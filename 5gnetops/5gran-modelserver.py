import os
import torch
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, AutoTokenizer
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and tokenizer
model_path = "models/5g_oss_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the model is on the correct device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return "GPT-2 5G Model Serving"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")
        
        input_data = data.get('input_data', '')
        question = data.get('question', '')

        if not input_data or not question:
            return jsonify({'error': 'Invalid input data or question'}), 400

        # Prepare the input text for the model
        input_text = f"Data: {input_data}\nQuestion: {question}\nAnswer:"

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        logging.debug(f"Tokenized inputs: {inputs}")

        # Generate predictions
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
            logging.debug(f"Model outputs: {outputs}")
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            return jsonify({'error': 'Error during model generation'}), 500

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Generated text: {generated_text}")

        # Extract the answer part
        answer = generated_text.split("Answer:")[-1].strip()

        return jsonify({'answer': answer})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
