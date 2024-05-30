import os
import torch
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, AutoTokenizer

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return "GPT-2 5G Model Serving"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = data.get('input_data', '')
    question = data.get('question', '')

    if not input_data or not question:
        return jsonify({'error': 'Invalid input data or question'}), 400

    # Prepare the input text for the model
    input_text = f"Data: {input_data}\nQuestion: {question}\nAnswer:"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt').to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer part
    answer = generated_text.split("Answer:")[-1].strip()

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
