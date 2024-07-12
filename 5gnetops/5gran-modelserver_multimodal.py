import os
import torch
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel, PeftConfig
import logging
import json

# Set environment variables to handle OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Model paths
MODEL_VERSION = "1.0.0"
models_dir = "models"
model_path = os.path.join(models_dir, "5gran_faultprediction_model")

# Load the trained models and tokenizer
xgb_model_path = os.path.join(model_path, f'xgboost_regressor_model_v{MODEL_VERSION}.joblib')
preprocessor_path = os.path.join(model_path, f'preprocessor_v{MODEL_VERSION}.joblib')
t5_model_path = os.path.join(model_path, f't5_model_v{MODEL_VERSION}')
tokenizer_path = os.path.join(model_path, f'tokenizer_v{MODEL_VERSION}')

# Load XGBoost model and preprocessor
xgb_model = joblib.load(xgb_model_path)
preprocessor = joblib.load(preprocessor_path)

# Get expected features from the preprocessor
expected_features = preprocessor.feature_names_in_

# Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# Load PEFT configuration
peft_config = PeftConfig.from_pretrained(t5_model_path)

# Load base T5 model
base_model = T5ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

# Resize token embeddings
base_model.resize_token_embeddings(len(tokenizer))

# Load the PEFT model
t5_model = PeftModel.from_pretrained(base_model, t5_model_path)

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

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

def prepare_t5_input(features_dict, prediction, threshold=30):
    severity = get_severity_level(prediction)
    above_below = "above" if prediction >= threshold else "below"
    input_text = f"Predicted Fault Occurrence Rate: {prediction}%, Severity: {severity}, Threshold: {threshold}% . "
    input_text += f"This is {above_below} the threshold of {threshold}% . "
    input_text += "Provide a concise explanation of the fault occurrence rate. Then, list three recommendations to improve wireless customer satisfaction. "
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

def generate_explanation(t5_input):
    input_ids = tokenizer(t5_input, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids.to(device)
    t5_model.eval()
    with torch.no_grad():
        outputs = t5_model.generate(
            input_ids=input_ids,
            max_length=300,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.debug(f"Generated explanation: {explanation}")
    return format_explanation(explanation)

def format_explanation(explanation):
    # Remove any existing newlines and extra spaces
    explanation = ' '.join(explanation.split())
    
    # Fix any split decimal points
    explanation = explanation.replace('. ', '.')
    
    # Split the explanation into sentences
    sentences = explanation.split('.')
    formatted_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:
            if "Recommendations" in sentence:
                formatted_sentences.append('\n' + sentence + '.')
            elif i > 0 and sentence[0].isdigit():
                formatted_sentences.append('\n' + sentence + '.')
            else:
                formatted_sentences.append(sentence + '.')
    
    # Join sentences and split recommendations
    formatted_text = ' '.join(formatted_sentences)
    parts = formatted_text.split('Recommendations')
    if len(parts) > 1:
        formatted_text = parts[0] + 'Recommendations' + parts[1]
    
    return formatted_text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")
        
        input_data = data.get('input_data', '')
        if not input_data:
            return jsonify({'error': 'Invalid input data'}), 400

        metrics = parse_input_metrics(input_data)
        input_df = prepare_input_dataframe(metrics)
        
        logging.debug(f"Input DataFrame:\n{input_df}")
        logging.debug(f"Input DataFrame dtypes:\n{input_df.dtypes}")
        
        input_preprocessed = preprocessor.transform(input_df)
        fault_rate = xgb_model.predict(input_preprocessed)[0]
        logging.debug(f"Predicted fault rate: {fault_rate}")

        severity = get_severity_level(fault_rate)
        t5_input = prepare_t5_input(metrics, round(fault_rate))
        explanation = generate_explanation(t5_input)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)