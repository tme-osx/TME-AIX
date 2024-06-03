# Author: Fatih E. NAR
# Note: Use Responsibly
#
import os
import torch
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and tokenizer
model_path = "models/5g_oss_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Ensure the model is on the correct device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return "T5-small 5G Model Serving"

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
        inputs = tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(device)
        logging.debug(f"Tokenized inputs: {inputs}")

        # Generate predictions
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=150,  # Increase max_length for more room to generate
                    num_return_sequences=1,
                    do_sample=True,  # Enable sampling
                    temperature=0.7,  # Lower temperature for less randomness
                    top_k=50,  # Consider only top k tokens
                    top_p=0.9  # Consider only top p cumulative probability
                )
            logging.debug(f"Model outputs: {outputs}")
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            return jsonify({'error': 'Error during model generation'}), 500

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Generated text: {generated_text}")

        # Extract the answer part
        answer = generated_text.split("Answer:")[-1].strip()

        # Post-process to remove repetitions and <pad> tokens
        answer = answer.replace('<pad>', '').strip()
        answer_sentences = answer.split('. ')
        unique_sentences = []
        for sentence in answer_sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
        answer = '. '.join(unique_sentences)

        return jsonify({'answer': answer})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
