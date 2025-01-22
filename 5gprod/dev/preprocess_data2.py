# Author: Fatih E. NAR
# 2025 Texas US
# This batch data preprocessing uses TransformerNN for Anomaly Detection.
# Please Go to -> https://platform.openai.com/settings/organization/api-keys and get your api-key
# And place in to "your_openai_apikey" below in line#30 this is to retrive VectorDB for your Metrics Datasets.
#
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import pickle
from tqdm import tqdm
import traceback

print("Starting preprocessing of 5G Core Network data...")
#OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_apikey"

class MetricsTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward*2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_feedforward, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Take the last sequence element
        return self.sigmoid(x).squeeze(-1)  # Squeeze to match target dimension

class MetricsDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

def load_data():
    """Load and prepare the raw data files."""
    print("Loading raw data...")
    try:
        amf_df = pd.read_csv('data2/amf_metrics.csv')
        smf_df = pd.read_csv('data2/smf_metrics.csv')
        upf_df = pd.read_csv('data2/upf_metrics.csv')
        
        for df in [amf_df, smf_df, upf_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
        
        with open('data2/alerts.json', 'r') as f:
            alerts = json.load(f)
        
        print("Data loaded successfully")
        return amf_df, smf_df, upf_df, alerts
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        raise

def prepare_sequences(df, labels, sequence_length=10, scaler=None):
    """Prepare sequences for the model training."""
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(df.select_dtypes(include=[np.number]).values)
    else:
        features = scaler.transform(df.select_dtypes(include=[np.number]).values)
    
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

def train_model(X_train, y_train, input_dim, accelerator):
    """Train the transformer model."""
    model = MetricsTransformer(input_dim).to(accelerator.device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Adjust batch size if needed
    batch_size = min(32, len(X_train))
    train_dataset = MetricsDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            batch_y = batch_y.float()
            
            try:
                loss = criterion(outputs, batch_y)
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                print(f"Outputs shape: {outputs.shape}, Targets shape: {batch_y.shape}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    return model

def detect_anomalies(df, model, scaler, accelerator, sequence_length=10, threshold=0.5):
    """Detect anomalies using the trained model."""
    features = df.select_dtypes(include=[np.number])
    feature_names = features.columns
    features_scaled = scaler.transform(features.values)
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(features_scaled) - sequence_length)):
            sequence = features_scaled[i:i + sequence_length]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(accelerator.device)
            anomaly_score = model(sequence_tensor).cpu().item()
            
            if anomaly_score > threshold:
                current_values = features.iloc[i + sequence_length]
                window_values = features.iloc[i:i + sequence_length]
                
                anomalies = {}
                for col in feature_names:
                    mean = window_values[col].mean()
                    std = window_values[col].std()
                    current = current_values[col]
                    
                    if abs(current - mean) > 2 * std:
                        anomalies[col] = {
                            'value': float(current),
                            'mean': float(mean),
                            'std': float(std),
                            'deviation_sigma': float(abs(current - mean) / std if std != 0 else 0),
                            'confidence': float(anomaly_score)
                        }
                
                if anomalies:
                    results.append({
                        'timestamp': df.iloc[i + sequence_length]['timestamp'],
                        'anomalies': anomalies,
                        'score': anomaly_score
                    })
    
    return results

def create_vector_stores(dfs, anomalies_dict, alerts):
    """Create and save vector stores for context retrieval."""
    print("Creating enhanced vector stores...")
    try:
        embeddings = OpenAIEmbeddings()
        vector_stores = {}
        
        for name, df in zip(['amf', 'smf', 'upf'], dfs):
            print(f"Processing {name.upper()} metrics...")
            
            anomalies = anomalies_dict[name]
            anomalies_by_time = {str(a['timestamp']): a for a in anomalies}
            
            descriptions = []
            for idx, row in df.iterrows():
                timestamp = str(row['timestamp'])
                
                # Basic metrics description
                desc = f"Time: {timestamp}\n"
                desc += "Metrics:\n"
                for col in df.columns:
                    if col != 'timestamp':
                        desc += f"{col}: {row[col]}\n"
                
                # Add anomaly information if exists
                if timestamp in anomalies_by_time:
                    anomaly = anomalies_by_time[timestamp]
                    desc += "\nAnomalies:\n"
                    for metric, details in anomaly['anomalies'].items():
                        desc += f"{metric} - Value: {details['value']}, Deviation: {details['deviation_sigma']}σ\n"
                
                descriptions.append(Document(page_content=desc))
            
            # Create and save vector store
            texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(descriptions)
            vector_stores[name] = FAISS.from_documents(texts, embeddings)
            vector_stores[name].save_local(f"processed_data2/{name}_vector_store")
            
            with open(f"processed_data2/{name}_documents.pkl", 'wb') as f:
                pickle.dump(descriptions, f)
        
        return vector_stores
    except Exception as e:
        print(f"Error creating vector stores: {str(e)}")
        traceback.print_exc()
        return {}

def perform_rca(timestamps, alerts, anomalies, vector_stores):
    """Perform root cause analysis."""
    print("Performing enhanced RCA...")
    try:
        llm = ChatOpenAI(temperature=0, model_name='gpt-4')
        rca_results = []
        
        def timestamp_handler(obj):
            """Custom JSON serializer for handling Timestamp objects."""
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

        for timestamp in tqdm(timestamps):
            active_alerts = [
                alert for alert in alerts['alerts']
                if (datetime.strptime(alert['start_time'], '%Y-%m-%d %H:%M:%S.%f') <= timestamp
                    and datetime.strptime(alert['end_time'], '%Y-%m-%d %H:%M:%S.%f') >= timestamp)
            ]
            
            # Get anomalies for current timestamp and convert timestamps to strings
            current_anomalies = {}
            for component in ['amf', 'smf', 'upf']:
                component_anomalies = []
                for anomaly in anomalies[component]:
                    if pd.Timestamp(anomaly['timestamp']) == timestamp:
                        # Create a copy of the anomaly dict and convert timestamp
                        processed_anomaly = anomaly.copy()
                        processed_anomaly['timestamp'] = anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        component_anomalies.append(processed_anomaly)
                current_anomalies[component] = component_anomalies
            
            if not active_alerts and not any(current_anomalies.values()):
                continue
            
            # Fetch relevant context from vector stores
            contexts = []
            for name, store in vector_stores.items():
                results = store.similarity_search(f"Find metrics and events at {timestamp}", k=2)
                contexts.extend([r.page_content for r in results])
            
            # Convert timestamp to string format for the prompt
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            prompt = f"""
            Analyze 5G Core network state at {timestamp_str}:
            
            Active Alerts: {json.dumps(active_alerts, indent=2, default=timestamp_handler)}
            Current Anomalies: {json.dumps(current_anomalies, indent=2, default=timestamp_handler)}
            Context: {' '.join(contexts)}
            
            Provide:
            1. Root Cause Analysis:
               - Primary issues and correlations
               - Impact assessment
               - Confidence level
            
            2. Recommendations:
               - Immediate actions
               - Prevention measures
               - Monitoring suggestions
            """
            
            try:
                analysis = llm.invoke(prompt)
                sections = str(analysis).split('\n\n')
                
                rca_results.append({
                    'timestamp': timestamp_str,
                    'rca': sections[0] if len(sections) > 0 else "No analysis available",
                    'recommendations': sections[1] if len(sections) > 1 else "No recommendations available"
                })
                
            except Exception as e:
                print(f"Error in RCA for {timestamp_str}: {str(e)}")
                continue
        
        return rca_results
    
    except Exception as e:
        print(f"Error in RCA process: {str(e)}")
        traceback.print_exc()
        return []


def main():
    """Main execution function."""
    try:
        # Initialize accelerator for GPU support
        accelerator = Accelerator()
        print(f"Using device: {accelerator.device}")
        
        # Create directories
        os.makedirs('processed_data2', exist_ok=True)
        
        # Load data
        amf_df, smf_df, upf_df, alerts = load_data()
        dfs = [amf_df, smf_df, upf_df]
        
        # Create labels from alerts
        labels = {}
        components = ['AMF', 'SMF', 'UPF']
        for df, component in zip(dfs, components):
            labels[component.lower()] = np.zeros(len(df))
            for alert in alerts['alerts']:
                if alert['component'] == component:
                    start = datetime.strptime(alert['start_time'], '%Y-%m-%d %H:%M:%S.%f')
                    end = datetime.strptime(alert['end_time'], '%Y-%m-%d %H:%M:%S.%f')
                    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
                    labels[component.lower()][mask] = 1
        
        # Initialize dictionaries
        models = {}
        scalers = {}
        anomalies = {'amf': [], 'smf': [], 'upf': []}  # Initialize with empty lists
        
        # Train models and detect anomalies
        for name, df in zip(['amf', 'smf', 'upf'], dfs):
            print(f"\nProcessing {name.upper()}...")
            try:
                X, y, scaler = prepare_sequences(df, labels[name])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
                model = train_model(X_train, y_train, X.shape[2], accelerator)
                models[name] = model
                scalers[name] = scaler
                
                # Save model and scaler
                torch.save(model.state_dict(), f'processed_data2/{name}_model.pth')
                with open(f'processed_data2/{name}_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                
                # Detect anomalies
                print(f"Detecting anomalies for {name}...")
                anomalies[name] = detect_anomalies(df, model, scaler, accelerator)
                print(f"Found {len(anomalies[name])} anomalies")
                
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Save anomalies even if some components failed
        print("\nSaving anomalies...")
        with open('processed_data2/anomalies.pkl', 'wb') as f:
            pickle.dump(anomalies, f)
        
        # Create vector stores if we have any anomalies
        try:
            print("\nCreating vector stores...")
            vector_stores = create_vector_stores(dfs, anomalies, alerts)
            
            # Perform RCA
            print("\nPerforming RCA...")
            all_timestamps = sorted(set(
                pd.concat([df['timestamp'] for df in dfs])
            ))
            rca_results = perform_rca(all_timestamps, alerts, anomalies, vector_stores)
            
            # Save RCA results
            print("\nSaving RCA results...")
            with open('processed_data2/rca_results.pkl', 'wb') as f:
                pickle.dump(rca_results, f)
                
            print(f"\nSummary of results:")
            print(f"- Total anomalies detected: {sum(len(anomalies[k]) for k in anomalies)}")
            print(f"- RCA analyses performed: {len(rca_results)}")
            
        except Exception as e:
            print(f"Error in post-processing: {str(e)}")
            traceback.print_exc()
        
        print("\nPreprocessing complete! All results saved in processed_data2/")
        
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
