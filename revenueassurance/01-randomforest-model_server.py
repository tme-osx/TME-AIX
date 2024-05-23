# Author: Fatih E. NAR
# Note: Use Responsibly
#
import pandas as pd
from flask import Flask, request, jsonify
import pickle

# Load the trained BalancedRandomForestClassifier model and feature names from the .pkl file
with open('revenueassurance/models/model2.pkl', 'rb') as model_file:
    model, feature_names = pickle.load(model_file)

# Ensure the loaded object is indeed a RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
