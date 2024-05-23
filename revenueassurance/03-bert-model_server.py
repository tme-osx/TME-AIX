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

# Softmax function to convert logits to probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract the required fields from the input data
        input_features = np.array([[
            int(data['Call_Duration']),
            int(data['Data_Usage']),
            int(data['Sms_Count']),
            int(data['Roaming_Indicator']),
            int(data['MobileWallet_Use']),
            1 if data['Plan_Type'] == 'postpaid' else 0,
            int(data['Cost']),
            int(data['Cellular_Location_Distance']),
            int(data['Last_Time_Pin_Used']),
            int(data['Avg_Call_Duration']),
            int(data['Avg_Data_Usage']),
            int(data['Avg_Cost'])
        ]], dtype=np.int64)  # Ensure the input features are in the correct format
        
        # Pad the input features to length 512
        padded_input_features = np.pad(input_features, ((0, 0), (0, 512 - input_features.shape[1])), 'constant', constant_values=0)
        
        # Prepare a default attention mask (all ones) with length 512
        attention_mask = np.ones((1, 512), dtype=np.int64)

        # Prepare the input dictionary for the ONNX model
        inputs = {
            session.get_inputs()[0].name: padded_input_features,
            session.get_inputs()[1].name: attention_mask
        }

        # Run inference
        outputs = session.run(None, inputs)
        probabilities = softmax(outputs[0])
        
        # Dynamic threshold from request or use default
        threshold = data.get('threshold', 0.5)
        
        prediction = "Fraud" if probabilities[0][1] >= threshold else "Non-Fraud"
        probability = probabilities[0][1] * 100

        # Log the probabilities for analysis
        print(f"Probabilities: {probabilities}, Threshold: {threshold}")

        return jsonify({"prediction": f"{prediction} with {probability:.3f}% probability", "probabilities": probabilities.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
