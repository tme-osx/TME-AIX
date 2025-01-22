# Author: Fatih E. NAR
# 2025 Texas US
# This batch data preprocessing uses Isolation Forest for Anomaly Detection.
#
# Please Go to -> https://platform.openai.com/settings/organization/api-keys and get your api-key
# And place in to "your_openai_apikey" below in line#25 this is to retrive VectorDB for your Metrics Datasets.
#
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import IsolationForest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import pickle
from tqdm import tqdm

print("Starting preprocessing of 5G Core Network data...")

# Get your api key from your embedding AIaaS endpoint and place below.
os.environ["OPENAI_API_KEY"] = "your_openai_apikey"

# Create directories for saved data
os.makedirs('processed_data', exist_ok=True)

# Load the datasets
def load_data():
    print("Loading raw data...")
    amf_df = pd.read_csv('data2/amf_metrics.csv')
    smf_df = pd.read_csv('data2/smf_metrics.csv')
    upf_df = pd.read_csv('data2/upf_metrics.csv')
    
    for df in [amf_df, smf_df, upf_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    
    with open('data2/alerts.json', 'r') as f:
        alerts = json.load(f)
    
    return amf_df, smf_df, upf_df, alerts

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name='gpt-4')

# Load and preprocess data
amf_df, smf_df, upf_df, alerts = load_data()

# Create vector stores
def create_vector_stores():
    print("Creating vector stores...")
    metrics_data = {
        'amf': amf_df,
        'smf': smf_df,
        'upf': upf_df
    }
    
    vector_stores = {}
    embeddings = OpenAIEmbeddings()
    
    for name, df in metrics_data.items():
        print(f"Processing {name.upper()} metrics...")
        df['description'] = df.apply(
            lambda row: f"{row['timestamp']} - " +
                       " | ".join([f"{col}: {row[col]}" for col in df.columns if col != 'timestamp']),
            axis=1
        )
        
        documents = [Document(page_content=row['description']) for _, row in df.iterrows()]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vector_stores[name] = FAISS.from_documents(texts, embeddings)
        
        # Save vector store
        vector_stores[name].save_local(f"processed_data/{name}_vector_store")
        
        # Save the raw documents for later use
        with open(f"processed_data/{name}_documents.pkl", 'wb') as f:
            pickle.dump(documents, f)
    
    return vector_stores

# Detect anomalies
def detect_anomalies_batch(df, window_size=30):
    print(f"Detecting anomalies for {df.shape[0]} timestamps...")
    results = []
    
    # Configure IsolationForest with more sensitive settings
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expect 10% of data points to be anomalous
        n_estimators=100,   # More trees for better accuracy
        random_state=42
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for i in tqdm(range(df.shape[0])):
        current_time = df.iloc[i]['timestamp']
        window_start = current_time - timedelta(minutes=window_size)
        df_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_time)]
        
        anomalies = {}
        for col in numeric_cols:
            if col == 'timestamp':
                continue
                
            try:
                data = df_window[col].values.reshape(-1, 1)
                
                # Skip if not enough data points
                if len(data) < 2:
                    continue
                    
                # Calculate statistical measures
                mean = np.mean(data)
                std = np.std(data)
                current_value = data[-1][0]
                
                # Detect anomalies using both IsolationForest and statistical thresholds
                predictions = isolation_forest.fit_predict(data)
                is_statistical_anomaly = abs(current_value - mean) > 3 * std
                is_isolation_anomaly = predictions[-1] == -1
                
                if is_isolation_anomaly or is_statistical_anomaly:
                    anomalies[col] = {
                        'value': float(current_value),
                        'mean': float(mean),
                        'std': float(std),
                        'deviation_sigma': float(abs(current_value - mean) / std if std != 0 else 0),
                        'detection_method': 'isolation_forest' if is_isolation_anomaly else 'statistical'
                    }
                    
            except Exception as e:
                print(f"Error detecting anomalies for {col} at {current_time}: {str(e)}")
                continue
        
        results.append({
            'timestamp': current_time,
            'anomalies': anomalies
        })
    
    return results

# Perform RCA for each timestamp
def perform_rca_batch(timestamps, alerts, anomalies, vector_stores):
    print("Performing batch RCA analysis...")
    rca_results = []
    
    # Convert anomalies to timestamp-indexed dictionary for faster lookup
    anomalies_by_component = {
        component: {
            pd.Timestamp(entry['timestamp']): entry['anomalies']
            for entry in anomalies[component]
        }
        for component in anomalies
    }

    # Process each timestamp at 5-minute intervals
    timestamps_to_analyze = [ts for ts in timestamps if ts.minute % 5 == 0]
    for timestamp in tqdm(timestamps_to_analyze):
        try:
            # Get active alerts for current timestamp
            active_alerts = [
                alert for alert in alerts['alerts']
                if (datetime.strptime(alert['start_time'], '%Y-%m-%d %H:%M:%S.%f') <= timestamp
                    and datetime.strptime(alert['end_time'], '%Y-%m-%d %H:%M:%S.%f') >= timestamp)
            ]
            
            # Get anomalies for current timestamp
            current_anomalies = {
                'AMF': anomalies_by_component['amf'].get(timestamp, {}),
                'SMF': anomalies_by_component['smf'].get(timestamp, {}),
                'UPF': anomalies_by_component['upf'].get(timestamp, {})
            }

            # Skip if no issues to analyze
            if not active_alerts and not any(anomalies for anomalies in current_anomalies.values()):
                rca_results.append({
                    'timestamp': timestamp,
                    'rca': "No active issues to analyze",
                    'recommendations': "No current recommendations"
                })
                continue

            # Fetch context from vector stores
            def fetch_context(vector_store, ts):
                query = f"Find metrics around {ts}"
                results = vector_store.similarity_search(query, k=2)
                return "\n".join([res.page_content for res in results])

            # Generate analysis prompt
            prompt = f"""
            Analyze the following 5G Core network state at {timestamp}:
            
            Active Alerts: {json.dumps(active_alerts, indent=2)}
            
            Metric Anomalies:
            AMF: {json.dumps(current_anomalies['AMF'], indent=2)}
            SMF: {json.dumps(current_anomalies['SMF'], indent=2)}
            UPF: {json.dumps(current_anomalies['UPF'], indent=2)}
            
            Context:
            AMF Metrics: {fetch_context(vector_stores['amf'], timestamp)}
            SMF Metrics: {fetch_context(vector_stores['smf'], timestamp)}
            UPF Metrics: {fetch_context(vector_stores['upf'], timestamp)}
            
            Based on the information provided:
            
            1. Identify the root cause:
               * Which components are primarily affected?
               * What specific metrics show abnormal behavior?
               * How are the issues potentially related?
               * What is the likely primary cause?
            
            2. Provide specific recommendations:
               * Immediate actions needed to address the issues
               * Long-term preventive measures
               * Additional monitoring suggestions
               * Steps to prevent recurrence
            
            Format your response in two clear sections:
            1. Root Cause Analysis:
               - Primary issues detected
               - Impact on network components
               - Correlation between alerts and metric anomalies
            2. Recommendations:
               - Immediate actions needed
               - Long-term preventive measures
               - Monitoring suggestions
            """

            # Get analysis from LLM
            analysis = llm.invoke(prompt)
            sections = str(analysis).split("2. Recommendations:")
            rca_text = sections[0].replace("1. Root Cause Analysis:", "").strip()
            recommendations = sections[1].strip() if len(sections) > 1 else "No specific recommendations"

            rca_results.append({
                'timestamp': timestamp,
                'rca': rca_text,
                'recommendations': recommendations
            })
            
        except Exception as e:
            print(f"Error analyzing timestamp {timestamp}: {str(e)}")
            rca_results.append({
                'timestamp': timestamp,
                'rca': f"Error performing analysis: {str(e)}",
                'recommendations': "No recommendations available due to analysis error"
            })
            continue

    return rca_results

def main():
    # Create vector stores
    vector_stores = create_vector_stores()
    
    # Detect anomalies for each component
    print("\nDetecting anomalies for each component...")
    anomalies = {
        'amf': detect_anomalies_batch(amf_df),
        'smf': detect_anomalies_batch(smf_df),
        'upf': detect_anomalies_batch(upf_df)
    }
    
    # Save anomalies
    print("\nSaving anomaly detection results...")
    with open('processed_data/anomalies.pkl', 'wb') as f:
        pickle.dump(anomalies, f)
    
    # Get unique timestamps from all dataframes
    all_timestamps = sorted(set(
        pd.concat([amf_df['timestamp'], smf_df['timestamp'], upf_df['timestamp']])
    ))
    
    # Perform batch RCA
    print("\nPerforming batch RCA analysis...")
    rca_results = perform_rca_batch(all_timestamps, alerts, anomalies, vector_stores)
    
    # Save RCA results
    print("\nSaving RCA results...")
    with open('processed_data/rca_results.pkl', 'wb') as f:
        pickle.dump(rca_results, f)
    
    print("\nPreprocessing complete! Results saved in processed_data/")

if __name__ == "__main__":
    main()
