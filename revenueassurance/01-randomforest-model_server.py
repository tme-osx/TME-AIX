# Author: Fatih E. NAR
# Note: Use Responsibly
#
import os
import threading
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pickle
from sklearn.ensemble import RandomForestClassifier

load_dotenv(override=True)

base_model = os.getenv("MODEL_TO_LOAD", "models/brfc_model.pkl")

# Load the trained BalancedRandomForestClassifier model and feature names from the .pkl file
with open(base_model, 'rb') as model_file:
    model, feature_names = pickle.load(model_file)

# Ensure the loaded object is indeed a RandomForestClassifier
assert isinstance(model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier"

# Initialize Flask application
app = Flask(__name__)

# Define a route for the default URL, which serves a simple welcome message
@app.route('/')
def home():
    return "Welcome to the Fraud Detection Model Server!"

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the JSON data to a DataFrame
    input_data = pd.DataFrame([data])
    
    # Ensure the columns match the training data
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    
    # Map the prediction to a more user-friendly response
    prediction_label = 'Fraud' if prediction[0] == 1 else 'Non-Fraud'
    
    # Return the prediction result as JSON
    return jsonify({'Prediction Result': f': {prediction_label}'})

# Function to run the Flask app in a separate thread
def run_app():
    app.run(debug=False, host='0.0.0.0', port=35000)

# Start the Flask app in a separate thread
if __name__ == '__main__':
    threading.Thread(target=run_app).start()
