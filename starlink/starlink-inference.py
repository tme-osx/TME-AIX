from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import torch
import joblib
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "max_age": 3600
    }
})

# Directory setup
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")

# Load data
try:
    with open(os.path.join(DATA_DIR, 'starlink_locations.json'), 'r') as f:
        locations_data = json.load(f)
    print(f"Loaded {len(locations_data)} locations")

    with open(os.path.join(DATA_DIR, 'elevation_cache.json'), 'r') as f:
        elevation_cache = json.load(f)
    print("Loaded elevation cache")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    locations_data = []
    elevation_cache = {}

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

# Global model variables
model = None
scaler = None
label_encoders = None

def load_model():
    global model, scaler, label_encoders
    try:
        model = StarlinkTransformer(input_dim=8)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'starlink_transformer.pth')))
        model.eval()
        
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.joblib'))
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Routes with CORS handling
@app.route('/')
def home():
    response = make_response(render_template('index.html'))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/locations', methods=['GET', 'OPTIONS'])
def get_locations():
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    response = make_response(jsonify(locations_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/elevation/<float:lat>/<float:lon>', methods=['GET', 'OPTIONS'])
def get_elevation(lat, lon):
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    key = f"{lat},{lon}"
    elevation = elevation_cache.get(key, 100.0)
    response = make_response(jsonify({'elevation': elevation}))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    if model is None and not load_model():
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        lat = data['latitude']
        lon = data['longitude']
        season = data['season']
        weather = data['weather']
        
        elevation = elevation_cache.get(f"{lat},{lon}", 100.0)
        
        features = np.array([[
            lat,
            lon,
            elevation,
            15,  # visible_satellites
            3,   # serving_satellites
            0.51, # signal_loss_db
            label_encoders['season'].transform([season])[0],
            label_encoders['weather'].transform([weather])[0]
        ]])
        
        features_scaled = scaler.transform(features)
        
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled)
            prediction = model(X.unsqueeze(1))
            qoe = prediction.item()
        
        response = make_response(jsonify({
            'qoe': round(qoe, 2),
            'satellites': 15,
            'download': round(130 * (qoe/100)),
            'upload': round(15 * (qoe/100)),
            'latency': round(40 + abs(lat)/90 * 20)
        }))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/cors', methods=['GET', 'OPTIONS'])
def debug_cors():
    """Debug endpoint to test CORS"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    return jsonify({
        'status': 'ok',
        'headers': dict(request.headers),
        'origin': request.headers.get('Origin', 'none'),
        'method': request.method
    })

def handle_preflight():
    """Handle CORS preflight requests"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    print("\nStarlink Performance Predictor")
    print("=============================")
    
    if load_model():
        print("Model loaded successfully")
    else:
        print("Warning: Model not loaded")
    
    print("\nAvailable endpoints:")
    print("1. Web Interface: http://localhost:35001/")
    print("2. API Endpoints:")
    print("   - GET  /locations")
    print("   - GET  /elevation/<lat>/<lon>")
    print("   - POST /predict")
    print("3. Debug: /debug/cors\n")
    
    app.run(host='0.0.0.0', port=35001, debug=False)