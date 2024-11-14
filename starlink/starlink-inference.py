from flask import Flask, request, jsonify, render_template
import torch
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Model definition
class StarlinkTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        self.fc = torch.nn.Linear(input_dim, 1)
        
    def forward(self, x):
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

# Load model and preprocessing objects
model = StarlinkTransformer(input_dim=8)
model.load_state_dict(torch.load('models/starlink_transformer.pth', weights_only=True))
model.eval()

scaler = joblib.load('models/scaler.joblib')
label_encoders = joblib.load('models/label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([[
            data['latitude'],
            data['longitude'],
            data['elevation'],
            15,  # visible_satellites default
            3,   # serving_satellites default
            0.51, # signal_loss_db default
            label_encoders['season'].transform([data['season']])[0],
            label_encoders['weather'].transform([data['weather']])[0]
        ]])
        
        features_scaled = scaler.transform(features)
        
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled)
            prediction = model(X.unsqueeze(1))
            qoe = prediction.item()
        
        return jsonify({
            'qoe': round(qoe, 2),
            'satellites': 15,
            'download': round(130 * (qoe/100)),
            'upload': round(15 * (qoe/100)),
            'latency': round(40 + abs(data['latitude'])/90 * 20)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=35001, debug=False)
