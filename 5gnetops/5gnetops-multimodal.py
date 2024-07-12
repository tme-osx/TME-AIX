# Multi-Modal Version of 5gran-predictions.
# Author: Fatih E. NAR (He is such a great guy with a great heart)
# NOTES: This work leverages x2 models ; XGBoost for Failure Rate Prediction -> Outputs Creates Fine-Tuning Data for T5-Base Model
#
import os
import shutil
import lzma
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, Seq2SeqTrainingArguments, __version__ as transformers_version
import torch
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour of the T5Tokenizer.")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize accelerator and set device
accelerator = Accelerator()
device = accelerator.device

# Set seed for reproducibility
set_seed(42)

# Model paths
MODEL_VERSION = "1.0.0"
models_dir = "models"
model_path = os.path.join(models_dir, "5gran_faultprediction_model")
model_name = "t5-base"  # Use "t5-small" for less powerful GPUs

# Create directories
os.makedirs(model_path, exist_ok=True)

# Check CUDA availability
fp16v = torch.cuda.is_available()

def extract_data(input_path, output_path):
    with lzma.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def load_data(data_path):
    return pd.read_csv(data_path)

def find_closest_column(df, target_column):
    return df.columns[df.columns.str.lower().str.contains(target_column.lower())].tolist()

def check_nan_and_inf(data):
    numeric_data = data.select_dtypes(include=[np.number])
    non_numeric_data = data.select_dtypes(exclude=[np.number])

    if numeric_data.isnull().values.any():
        logging.warning("NaN values found in numeric data!")
        numeric_data = numeric_data.dropna()
    if np.isinf(numeric_data.values).any():
        logging.warning("Infinite values found in numeric data!")
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()

    if non_numeric_data.isnull().values.any():
        logging.warning("NaN values found in non-numeric data!")
        non_numeric_data = non_numeric_data.dropna()

    return pd.concat([numeric_data, non_numeric_data], axis=1)

def preprocess_data(data, initial_features, target):
    data = check_nan_and_inf(data)
    features = []
    for feature in initial_features:
        matches = find_closest_column(data, feature)
        if matches:
            features.extend(matches)
        else:
            logging.warning(f"No match found for '{feature}'")

    logging.info(f"Updated features list: {features}")

    target_matches = find_closest_column(data, target)
    if target_matches:
        target = target_matches[0]
        logging.info(f"Target column: {target}")
    else:
        raise ValueError(f"Cannot find the target column '{target}'")

    return features, target

def create_preprocessor(numeric_features, categorical_features):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('poly', poly, numeric_features),
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor

def get_severity_level(fault_rate):
    if 0 <= fault_rate < 10:
        return "Very Low"
    elif 10 <= fault_rate < 20:
        return "Low"
    elif 20 <= fault_rate < 30:
        return "Moderate-Low"
    elif 30 <= fault_rate < 40:
        return "Moderate"
    elif 40 <= fault_rate < 50:
        return "Moderate-High"
    elif 50 <= fault_rate < 60:
        return "High"
    elif 60 <= fault_rate < 70:
        return "Very High"
    elif 70 <= fault_rate < 80:
        return "Critical"
    elif 80 <= fault_rate < 90:
        return "Very Critical"
    else:
        return "Extreme"

def prepare_t5_input(features_dict, prediction, threshold=30):
    input_text = f"Predicted Fault Occurrence Rate: {prediction:.2f}%, Threshold: {threshold}%. "
    input_text += "Provide an explanation of the fault occurrence rate and suggest actions to improve wireless customer satisfaction. "
    input_text += "Key metrics: " + ", ".join([f"{name}: {value}" for name, value in features_dict.items()])
    return input_text

def generate_custom_explanation(features, fault_rate, severity, threshold):
    explanation = f"The predicted fault occurrence rate of {fault_rate:.2f}% is in the {severity} range. "
    if fault_rate < threshold:
        explanation += f"This is below the threshold of {threshold}%, suggesting relatively stable network performance. "
    else:
        explanation += f"This is above the threshold of {threshold}%, indicating potential network issues that need attention. "

    explanation += f"Recommendations to improve network performance and customer satisfaction:\n"

    if severity == "Very Low":
        explanation += "1. Maintain current network optimization strategies.\n"
        explanation += "2. Focus on proactive maintenance to prevent future issues.\n"
        explanation += "3. Explore opportunities for enhancing user experience beyond basic connectivity.\n"
    elif severity == "Low":
        explanation += "1. Review and optimize network capacity during peak hours.\n"
        explanation += "2. Implement predictive maintenance protocols.\n"
        explanation += "3. Conduct user satisfaction surveys to identify areas for improvement.\n"
    elif severity == "Moderate-Low":
        explanation += "1. Increase frequency of network performance monitoring.\n"
        explanation += "2. Investigate and address any recurring minor issues.\n"
        explanation += "3. Enhance customer support to quickly resolve user-reported problems.\n"
    elif severity == "Moderate":
        explanation += "1. Conduct a thorough analysis of network weak points.\n"
        explanation += "2. Implement more robust fault detection and prevention measures.\n"
        explanation += "3. Consider targeted infrastructure upgrades in problem areas.\n"
    elif severity == "Moderate-High":
        explanation += "1. Prioritize addressing high-impact factors such as cell availability and latency.\n"
        explanation += "2. Implement advanced network monitoring and automated alert systems.\n"
        explanation += "3. Develop a clear action plan for rapid response to emerging issues.\n"
    elif severity == "High":
        explanation += "1. Initiate emergency response protocols to mitigate current issues.\n"
        explanation += "2. Conduct an in-depth review of network architecture and identify major pain points.\n"
        explanation += "3. Prepare for potential temporary capacity expansions in affected areas.\n"
    elif severity == "Very High":
        explanation += "1. Activate crisis management team to address urgent network problems.\n"
        explanation += "2. Implement immediate measures to offload traffic from problematic nodes.\n"
        explanation += "3. Communicate transparently with customers about ongoing issues and improvement efforts.\n"
    elif severity == "Critical":
        explanation += "1. Escalate to highest priority - all hands on deck to resolve critical network failures.\n"
        explanation += "2. Consider temporary shutdown of non-essential services to prioritize core network functions.\n"
        explanation += "3. Prepare detailed reports for regulatory bodies and key stakeholders.\n"
    elif severity == "Very Critical":
        explanation += "1. Initiate emergency network reconfiguration to isolate and contain severe issues.\n"
        explanation += "2. Deploy all available resources, including third-party experts, to address the crisis.\n"
        explanation += "3. Establish a war room for continuous monitoring and rapid decision-making.\n"
    else:  # Extreme
        explanation += "1. Declare network emergency and potentially initiate partial or full network reset.\n"
        explanation += "2. Engage with government and regulatory bodies for potential support and guidance.\n"
        explanation += "3. Prepare for potential long-term reputational damage control and customer retention strategies.\n"

    return explanation

def create_custom_dataset(X, y, threshold=30):
    dataset = []
    for features, target in zip(X.to_dict('records'), y):
        severity = get_severity_level(target)  # Use actual severity based on target
        input_text = prepare_t5_input(features, target, threshold)
        output_text = generate_custom_explanation(features, target, severity, threshold)
        dataset.append({"input_text": input_text, "target_text": output_text})
    return dataset

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, features):
    logging.info("Starting XGBoost training and evaluation")

    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    xgb_model = xgb.XGBRegressor(tree_method='hist', device='cpu', random_state=42)
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"XGBoost Regressor MSE: {mse}")
    logging.info(f"XGBoost Regressor R2: {r2}")
    logging.info(f"XGBoost Regressor MAE: {mae}")

    feature_importance = best_model.feature_importances_
    feature_importance_dict = dict(zip(features, feature_importance))
    logging.info(f"Feature Importance: {feature_importance_dict}")

    return best_model

def predict_and_explain(input_data, xgb_model, preprocessor, t5_model, tokenizer, features, threshold=30):
    try:
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=features)

        input_preprocessed = preprocessor.transform(input_data)
        fault_rate = xgb_model.predict(input_preprocessed)[0]

        # Determine severity based on fault_rate
        severity = get_severity_level(fault_rate)

        t5_input = prepare_t5_input(input_data.iloc[0].to_dict(), fault_rate, threshold)

        input_ids = tokenizer(t5_input, return_tensors="pt", max_length=512, padding="max_length", truncation=True).input_ids.to(device)

        t5_model.to(device)
        t5_model.eval()

        with torch.no_grad():
            outputs = t5_model.generate(
                input_ids=input_ids,
                max_length=300,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        status = "Good" if fault_rate < threshold else "Concerning"

        # Post-process the explanation
        explanation_lines = explanation.split('\n')
        filtered_explanation = '\n'.join([line for line in explanation_lines if not line.startswith('Predicted Fault Occurrence Rate') and not line.startswith('Threshold')])

        final_output = f"Predicted Fault Occurrence Rate: {fault_rate:.2f}%\nStatus: {status}\nSeverity: {severity}\n\nExplanation and Recommendations:\n{filtered_explanation}"

        return final_output
    except Exception as e:
        logging.error(f"Error in predict_and_explain: {str(e)}")
        return f"An error occurred: {str(e)}"

# Main Execution
if __name__ == "__main__":
    # Step 1: Load and preprocess data
    logging.info("Step 1: Loading and preprocessing data...")
    extract_data('data/5G_netops_data_100K.csv.xz', 'data/5G_netops_data_100K.csv')

    data_path = "data/5G_netops_data_100K.csv"
    data = load_data(data_path)
    initial_features = [
        'Cell Availability', 'MTTR', 'Throughput', 'Latency', 'Packet Loss Rate', 'Call Drop Rate',
        'Handover Success Rate', 'Data Usage', 'User Count', 'Signal Strength', 'Jitter',
        'Connection Setup Success Rate', 'Security Incidents', 'Authentication Failures',
        'Temperature', 'Humidity', 'Weather'
    ]
    target = 'Fault Occurrence Rate'

    features, target = preprocess_data(data, initial_features, target)
    numeric_features = data[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data[features].select_dtypes(include=['object']).columns

    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")

    # Step 2: Create preprocessing steps
    logging.info("Step 2: Creating preprocessing steps...")
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    # Step 3: Split the data
    logging.info("Step 3: Splitting the data...")
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"Shape of X_train: {X_train.shape}")
    logging.info(f"Shape of y_train: {y_train.shape}")

    # Step 4: Preprocess the data
    logging.info("Step 4: Preprocessing the data...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Step 5: Train XGBoost Regressor
    logging.info("Step 5: Training XGBoost Regressor...")
    xgb_model = train_and_evaluate_xgboost(X_train_preprocessed, y_train, X_test_preprocessed, y_test, features)

    # Save the XGBoost model and preprocessor with version
    joblib.dump(xgb_model, os.path.join(model_path, f'xgboost_regressor_model_v{MODEL_VERSION}.joblib'))
    joblib.dump(preprocessor, os.path.join(model_path, f'preprocessor_v{MODEL_VERSION}.joblib'))

    # Step 6: Prepare T5 input data
    logging.info("Step 6: Preparing T5 input data...")

    # Load the tokenizer from the pretrained model
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)

    logging.info(f"Tokenizer length: {len(tokenizer)}")

    # Save the tokenizer with version
    tokenizer.save_pretrained(os.path.join(model_path, f'tokenizer_v{MODEL_VERSION}'))

    train_dataset = Dataset.from_list(create_custom_dataset(X_train, y_train))
    eval_dataset = Dataset.from_list(create_custom_dataset(X_test, y_test))

    # Step 7: Fine-tune T5 model
    logging.info("Step 7: Fine-tuning T5 model...")

    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_model.resize_token_embeddings(len(tokenizer))

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    columns = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format(type='torch', columns=columns)
    eval_dataset.set_format(type='torch', columns=columns)

    # Define PEFT/LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q', 'v', 'k', 'o', 'wi', 'wo']
    )
    t5_model = get_peft_model(t5_model, lora_config)

    # Prepare the model with Accelerator
    t5_model = accelerator.prepare(t5_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=26,
        gradient_accumulation_steps=14,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        save_steps=100,
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        predict_with_generate=True,
        fp16=fp16v,
        remove_unused_columns=False,
        warmup_steps=100,
        weight_decay=0.02,
    )

    trainer = Trainer(
        model=t5_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Save the T5 model with version
    t5_model.save_pretrained(os.path.join(model_path, f't5_model_v{MODEL_VERSION}'), save_embedding_layers=True)

    # Step 8: Generate sample prediction and explanation
    logging.info("Step 8: Generating sample prediction and explanation...")
    sample_input = X_test.iloc[0]
    try:
        result = predict_and_explain(sample_input, xgb_model, preprocessor, t5_model, tokenizer, features)
        logging.info(f"Sample prediction result:\n{result}")
        print(result)
    except Exception as e:
        logging.error(f"Error in sample prediction: {str(e)}")

    logging.info(f"Process completed successfully! Model version: {MODEL_VERSION}")
