# Multi-Modal Version of 5gran-predictions-model-server.
# Author: Fatih E. NAR (He is such a great guy with a great heart)
# NOTES: This work leverages x2 models; XGBoost for Failure Rate Prediction -> Outputs used for Querying Fine-Tuned BERT Model

import os
import torch
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import json

# Set environment variables to handle OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Model paths
MODEL_VERSION = "1.0.3"
models_dir = "../models"
model_path = os.path.join(models_dir, "5gran_faultprediction_model")
model_name = "bert-base-uncased"

# Load the trained models and tokenizer
xgb_model_path = os.path.join(model_path, f'xgboost_regressor_model_v{MODEL_VERSION}.joblib')
preprocessor_path = os.path.join(model_path, f'preprocessor_v{MODEL_VERSION}.joblib')
bert_model_path = os.path.join(model_path, f'bert_model_v{MODEL_VERSION}')
tokenizer_path = os.path.join(model_path, f'tokenizer_v{MODEL_VERSION}')

# Load XGBoost model and preprocessor
xgb_model = joblib.load(xgb_model_path)
preprocessor = joblib.load(preprocessor_path)

# Get expected features from the preprocessor
expected_features = preprocessor.feature_names_in_

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

logging.info(f"BERT model loaded. Number of parameters: {sum(p.numel() for p in bert_model.parameters())}")

def get_severity_level(fault_rate):
    levels = [
        (0, 10, "Very Low"),
        (10, 20, "Low"),
        (20, 30, "Moderate-Low"),
        (30, 40, "Moderate"),
        (40, 50, "Moderate-High"),
        (50, 60, "High"),
        (60, 70, "Very High"),
        (70, 80, "Critical"),
        (80, 90, "Very Critical"),
        (90, 100, "Extreme")
    ]
    for low, high, level in levels:
        if low <= fault_rate < high:
            return level
    return "Extreme"

def prepare_bert_input(features_dict, prediction, threshold=30):
    input_text = f"Predicted Fault Occurrence Rate: {prediction:.2f}%, Threshold: {threshold}%. "
    input_text += "Provide an explanation of the fault occurrence rate and suggest actions to improve wireless customer satisfaction. "
    input_text += "Key metrics: " + ", ".join([f"{name}: {value}" for name, value in features_dict.items()])
    return input_text

@app.route('/')
def home():
    return "5G Fault Prediction and Explanation Model Serving"

def parse_input_metrics(input_data):
    metrics = {}
    for item in input_data.split(','):
        key, value = item.split(':')
        key = key.strip()
        value = value.strip().rstrip('%').rstrip('Mbps').rstrip('ms').rstrip('dBm').rstrip('°C')
        if key == 'MTTR':
            value = float(value.split()[0])
        elif key == 'Data Usage':
            value = float(value.rstrip('G')) * 1000  # Convert GB to MB
        elif key not in ['Weather']:
            try:
                value = float(value)
            except ValueError:
                pass
        metrics[key] = value
    return metrics

def prepare_input_dataframe(metrics):
    input_df = pd.DataFrame(columns=expected_features)
    input_df.loc[0] = pd.NA
    for key, value in metrics.items():
        matched_columns = [col for col in expected_features if key.lower() in col.lower()]
        if matched_columns:
            input_df.loc[0, matched_columns[0]] = value

    # Convert numeric columns to float, ignoring errors
    numeric_columns = input_df.columns.drop(preprocessor.named_transformers_['cat'].feature_names_in_)
    input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with 0 for numeric columns
    input_df[numeric_columns] = input_df[numeric_columns].fillna(0)

    # Handle categorical features
    for col in preprocessor.named_transformers_['cat'].feature_names_in_:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')
        else:
            input_df[col] = pd.Categorical(['unknown'])

    return input_df

def predict_and_explain(input_data, xgb_model, preprocessor, bert_model, tokenizer, features, threshold=30):
    try:
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=features)

        input_preprocessed = preprocessor.transform(input_data)
        fault_rate = xgb_model.predict(input_preprocessed)[0]

        # Determine severity based on fault_rate
        severity = get_severity_level(fault_rate)

        bert_input = prepare_bert_input(input_data.iloc[0].to_dict(), fault_rate, threshold)

        inputs = tokenizer(bert_input, return_tensors="pt", max_length=512, padding="max_length", truncation=True).to(device)

        bert_model.to(device)
        bert_model.eval()

        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        explanation = generate_custom_explanation(input_data.iloc[0].to_dict(), fault_rate, severity, threshold)

        status = "Good" if fault_rate < threshold else "Concerning"

        final_output = f"Predicted Fault Occurrence Rate: {fault_rate:.2f}%\nStatus: {status}\nSeverity: {severity}\n\nExplanation and Recommendations:\n{explanation}"

        return final_output, fault_rate, severity
    except Exception as e:
        logging.error(f"Error in predict_and_explain: {str(e)}")
        return f"An error occurred: {str(e)}", None, None

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
    else:  # Extreme
        explanation += "1. Declare network emergency and potentially initiate partial or full network reset.\n"
        explanation += "2. Engage with government and regulatory bodies for potential support and guidance.\n"
        explanation += "3. Prepare for potential long-term reputational damage control and customer retention strategies.\n"

    return explanation

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        input_data = data.get('input_data', '')
        if not input_data:
            return jsonify({'error': 'Invalid input data'}), 400

        metrics = parse_input_metrics(input_data)
        logging.debug(f"Parsed metrics: {metrics}")

        input_df = prepare_input_dataframe(metrics)
        logging.debug(f"Prepared input DataFrame:\n{input_df}")

        explanation, fault_rate, severity = predict_and_explain(input_df, xgb_model, preprocessor, bert_model, tokenizer, expected_features)
        logging.debug(f"Generated explanation: {explanation}")

        response_data = {
            'fault_rate': float(fault_rate),
            'status': "Good" if fault_rate < 30 else "Concerning",
            'severity': severity,
            'explanation': explanation
        }

        return Response(
            json.dumps(response_data, indent=2, ensure_ascii=False),
            mimetype='application/json'
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

def test_bert_model():
    test_input = "Generate an explanation for a 5G network with high fault rate."
    inputs = tokenizer(test_input, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    result = tokenizer.decode(outputs.logits[0], skip_special_tokens=True)
    logging.info(f"BERT model test output: {result}")

# Add this to your main block
if __name__ == '__main__':
    test_bert_model()
    app.run(host='0.0.0.0', port=5000)
