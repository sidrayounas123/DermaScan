from flask import Flask, jsonify, request
import os
import json
from datetime import datetime
import threading
import time

# Import Firebase services
try:
    from app.firebase_service import save_scan, get_scans
    from app.auth_service import register_user, login_user, get_user_profile
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Import ML services
try:
    from app.model1 import load_model1, predict1, CLASS_NAMES_1
    from app.model2 import load_model2, predict2, CLASS_NAMES_2
    from app.utils import preprocess_image
    from app.disease_info import DISEASE_INFO
    ML_AVAILABLE = True
    MODELS_LOADED = False
except ImportError:
    ML_AVAILABLE = False
    MODELS_LOADED = False

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def health_check():
    """Simple health check that responds immediately"""
    return jsonify({
        "message": "Backend is running",
        "status": "healthy",
        "service": "skin-disease-backend"
    })

@app.route('/status')
def status():
    """Status endpoint"""
    return jsonify({
        "status": "running",
        "mode": "complete-api",
        "firebase_available": FIREBASE_AVAILABLE,
        "ml_available": ML_AVAILABLE,
        "models_loaded": MODELS_LOADED,
        "message": "Complete backend functionality available"
    })

def load_models_in_background():
    """Load ML models in background thread"""
    global MODELS_LOADED
    if ML_AVAILABLE and not MODELS_LOADED:
        try:
            print("Loading ML models in background...")
            load_model1()
            load_model2()
            MODELS_LOADED = True
            print("ML models loaded successfully!")
        except Exception as e:
            print(f"Failed to load ML models: {e}")
            MODELS_LOADED = False

# Start model loading in background
if ML_AVAILABLE:
    model_thread = threading.Thread(target=load_models_in_background, daemon=True)
    model_thread.start()

# Authentication endpoints
@app.route('/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({"error": "Missing required fields"}), 400
        
        result = register_user(name, email, password)
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({"error": "Missing email or password"}), 400
        
        result = login_user(email, password)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/profile/<uid>', methods=['GET'])
def get_profile(uid):
    """Get user profile endpoint"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        result = get_user_profile(uid)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Scan history endpoint
@app.route('/scans/<user_id>', methods=['GET'])
def get_user_scans(user_id):
    """Get user scan history"""
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase services not available"}), 503
    
    try:
        scans = get_scans(user_id)
        return jsonify({"scans": scans}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Prediction endpoints with real ML functionality
@app.route('/predict/dataset1', methods=['POST'])
def predict_dataset1():
    """Predict skin disease using Model 1 (Dataset 1)"""
    if not ML_AVAILABLE:
        return jsonify({"error": "ML modules not available"}), 503
    
    if not MODELS_LOADED:
        return jsonify({"error": "ML models still loading, please try again in 30 seconds"}), 503
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type. Only images (jpg, png) are allowed."}), 400
        
        # Check if model file exists
        if not os.path.exists("weights/model1.pth"):
            return jsonify({"error": "Model 1 not available - weights/model1.pth not found"}), 503
        
        # Get user_id from query params
        user_id = request.args.get('user_id')
        
        # Read and preprocess image
        image_bytes = file.read()
        image = preprocess_image(image_bytes)
        
        # Make prediction
        disease, confidence = predict1(image)
        
        # Get disease information
        info = DISEASE_INFO.get(disease, {
            "severity": "Unknown",
            "see_doctor": True,
            "description": "No description available"
        })
        
        # Save to Firebase if user_id provided
        if user_id and FIREBASE_AVAILABLE:
            try:
                save_scan(user_id, disease, confidence, info["severity"], info["see_doctor"], "dataset1")
            except Exception as e:
                print(f"Failed to save scan: {e}")
        
        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "severity": info["severity"],
            "see_doctor": info["see_doctor"],
            "description": info["description"],
            "dataset": "dataset1"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/dataset2', methods=['POST'])
def predict_dataset2():
    """Predict skin disease using Model 2 (Dataset 2)"""
    if not ML_AVAILABLE:
        return jsonify({"error": "ML modules not available"}), 503
    
    if not MODELS_LOADED:
        return jsonify({"error": "ML models still loading, please try again in 30 seconds"}), 503
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type. Only images (jpg, png) are allowed."}), 400
        
        # Check if classes are configured
        if len(CLASS_NAMES_2) == 0:
            return jsonify({"error": "Model 2 classes not configured"}), 503
        
        # Check if model file exists
        if not os.path.exists("weights/model2.pth"):
            return jsonify({"error": "Model 2 not available - weights/model2.pth not found"}), 503
        
        # Get user_id from query params
        user_id = request.args.get('user_id')
        
        # Read and preprocess image
        image_bytes = file.read()
        image = preprocess_image(image_bytes)
        
        # Make prediction
        disease, confidence = predict2(image)
        
        # Get disease information
        info = DISEASE_INFO.get(disease, {
            "severity": "Unknown",
            "see_doctor": True,
            "description": "No description available"
        })
        
        # Save to Firebase if user_id provided
        if user_id and FIREBASE_AVAILABLE:
            try:
                save_scan(user_id, disease, confidence, info["severity"], info["see_doctor"], "dataset2")
            except Exception as e:
                print(f"Failed to save scan: {e}")
        
        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "severity": info["severity"],
            "see_doctor": info["see_doctor"],
            "description": info["description"],
            "dataset": "dataset2"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Start Flask app
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
