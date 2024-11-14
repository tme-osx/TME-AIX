from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import joblib
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global variables
model = None
scaler = None
label_encoders = None

# Get the current working directory for Jupyter
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

print(f"Base directory: {BASE_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Templates directory: {TEMPLATES_DIR}")

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

def load_model():
    """Load model and preprocessing objects"""
    global model, scaler, label_encoders
    
    try:
        model_path = os.path.join(MODELS_DIR, 'starlink_transformer.pth')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        encoders_path = os.path.join(MODELS_DIR, 'label_encoders.joblib')
        
        print("\nChecking model files:")
        print(f"Model path exists: {os.path.exists(model_path)}")
        print(f"Scaler path exists: {os.path.exists(scaler_path)}")
        print(f"Encoders path exists: {os.path.exists(encoders_path)}")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
            raise FileNotFoundError(f"Missing model files in {MODELS_DIR}")
        
        # Load model
        model = StarlinkTransformer(input_dim=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Load preprocessing objects
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        
        print("Model and preprocessing objects loaded successfully")
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    try:
        template_path = os.path.join(TEMPLATES_DIR, 'index.html')
        print(f"Template path: {template_path}")
        print(f"Template exists: {os.path.exists(template_path)}")
        
        if not os.path.exists(template_path):
            return "Template not found", 404
            
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return str(e), 500

@app.route('/api')
def api_info():
    return jsonify({
        'endpoints': {
            '/': 'Web interface',
            '/api': 'This API documentation',
            '/predict': 'POST endpoint for predictions',
            '/health': 'Health check'
        },
        'predict_example': {
            'method': 'POST',
            'content-type': 'application/json',
            'body': {
                'latitude': 33.87,
                'longitude': -98.59,
                'elevation': 307,
                'season': 'Summer',
                'weather': 'Clear'
            }
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'cwd': os.getcwd(),
        'models_dir': MODELS_DIR,
        'templates_dir': TEMPLATES_DIR,
        'model_loaded': model is not None,
        'preprocessors_loaded': scaler is not None and label_encoders is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    global model, scaler, label_encoders
    
    try:
        if model is None or scaler is None or label_encoders is None:
            if not load_model():
                return jsonify({'error': 'Model not loaded properly'}), 500
        
        data = request.json
        print(f"Received request data: {data}")
        
        # Ensure all required fields are present
        required_fields = ['latitude', 'longitude', 'elevation', 'season', 'weather']
        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            return jsonify({'error': f'Missing required fields: {missing}'}), 400
        
        # Transform categorical variables
        try:
            season_encoded = label_encoders['season'].transform([data['season']])[0]
            weather_encoded = label_encoders['weather'].transform([data['weather']])[0]
        except ValueError as e:
            return jsonify({'error': f'Invalid season or weather value: {str(e)}'}), 400
        
        # Prepare features
        features = np.array([[
            data['latitude'],
            data['longitude'],
            data['elevation'],
            15,  # visible_satellites default
            3,   # serving_satellites default
            0.51, # signal_loss_db default
            season_encoded,
            weather_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
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
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    
    # Set Flask template folder
    app.template_folder = TEMPLATES_DIR
    
    print("\nStarlink Predictor Server")
    print("=========================")
    print("\nDirectory Structure:")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"TEMPLATES_DIR: {TEMPLATES_DIR}")
    
    # Initial model load
    if load_model():
        print("\nModel loaded successfully")
    else:
        print("\nWarning: Model not loaded. Please check the model files location")
        print(f"Expected model files in: {MODELS_DIR}")
        print("Required files:")
        print("- starlink_transformer.pth")
        print("- scaler.joblib")
        print("- label_encoders.joblib")
    
    print("\nAvailable endpoints:")
    print("1. Web Interface: http://localhost:35001/")
    print("2. API Info:      http://localhost:35001/api")
    print("3. Health Check:  http://localhost:35001/health")
    print("4. Predictions:   http://localhost:35001/predict (POST)\n")
    
    app.run(host='0.0.0.0', port=35001, debug=False)