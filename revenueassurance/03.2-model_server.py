# Model Server for BERTwith PEFT/LoRA for Telco Revenue Assurance Fraud Detection
# Author: Fatih E. NAR
# PreReq: Python 3.8.9 with ssl support + pip install onnxruntime flask flask-cors
#
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the ONNX model
model_path = 'models/fine-tuned-bert-revass.onnx'
session = ort.InferenceSession(model_path)

# Print model input information
input_info = session.get_inputs()
for i, input in enumerate(input_info):
    print(f"Input {i}: name={input.name}, shape={input.shape}, type={input.type}")

# Function to pad sequences to the required length
def pad_sequences(sequences, max_length, pad_value=0):
    return np.array([seq + [pad_value] * (max_length - len(seq)) for seq in sequences], dtype=np.int64)

# Softmax function to convert logits to probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        max_length = 512  # Expected sequence length

        # Pad input sequences
        input_ids = pad_sequences(data['input_ids'], max_length)
        attention_mask = pad_sequences(data['attention_mask'], max_length)
        
        # Prepare the input dictionary for the ONNX model
        inputs = {
            session.get_inputs()[0].name: input_ids,
            session.get_inputs()[1].name: attention_mask
        }
        
        # If the model expects 'token_type_ids', include them as well
        if len(session.get_inputs()) > 2 and 'token_type_ids' in data:
            token_type_ids = pad_sequences(data['token_type_ids'], max_length)
            inputs[session.get_inputs()[2].name] = token_type_ids
        
        # Run inference
        outputs = session.run(None, inputs)
        
        # Postprocess the output data
        logits = outputs[0]
        probabilities = softmax(logits).tolist()
        predicted_classes = np.argmax(logits, axis=-1).tolist()
        
        # Map predicted classes to descriptive labels
        class_labels = ['Non-Fraud', 'Fraud']
        predicted_label = class_labels[predicted_classes[0]]
        predicted_probability = probabilities[0][predicted_classes[0]]
        
        return jsonify({'prediction': f'{predicted_label} with {predicted_probability * 100:.3f}% probability'})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)