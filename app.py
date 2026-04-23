from flask import Flask, jsonify
import os

app = Flask(__name__)

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
        "mode": "api-only",
        "message": "FastAPI app running in background"
    })

if __name__ == '__main__':
    # Start Flask app
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
