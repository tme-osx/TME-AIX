# Author: Fatih E. NAR
# Note: Use Responsibly
#
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

# Define a simple transformer-based model for tabular data
class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, d_model=32, nhead=8):
        super(SimpleTransformerModel, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x.unsqueeze(1))  # Add sequence dimension
        x = x.mean(dim=1)  # Aggregate over the sequence dimension, retaining batch size
        return self.fc(x)

# Load the pre-trained model
input_dim = 13  # Updated based on the actual number of input features
output_dim = 1  # Update to match the pre-trained model's output dimension
model = SimpleTransformerModel(input_dim, output_dim)
model.load_state_dict(torch.load("models/revass_transformer_model.pth"))  # Adjust path as necessary
model.eval()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['Call_Duration', 'Data_Usage', 'Sms_Count', 'Roaming_Indicator',
                                        'MobileWallet_Use', 'Cost', 'Cellular_Location_Distance',
                                        'Personal_Pin_Used', 'Avg_Call_Duration', 'Avg_Data_Usage', 'Avg_Cost']),
        ('cat', categorical_transformer, ['Plan_Type'])
    ])

# Fit the preprocessor with a sample data
# This should be the same data or a subset of the data that was used to fit the model during training
df = pd.read_csv('data/telecom_revass_data.csv')
preprocessor.fit(df.drop(columns=['Fraud']))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame([input_data])
    
    # Apply preprocessing
    input_data_processed = preprocessor.transform(df)
    
    # Convert input data to tensor
    input_tensor = torch.tensor(input_data_processed, dtype=torch.float32)
    
    # Make predictions using the loaded model
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor)).numpy()
    
    # Map the prediction to a more user-friendly response
    prediction_label = 'Fraud' if prediction[0] > 0.5 else 'Non-Fraud'
    
    # Return the prediction result as JSON
    return jsonify({'prediction': prediction_label})

# Function to run the Flask app in a separate thread
def run_app():
    app.run(debug=False, host='0.0.0.0', port=35000)

# Start the Flask app in a separate thread
if __name__ == '__main__':
    threading.Thread(target=run_app).start()
