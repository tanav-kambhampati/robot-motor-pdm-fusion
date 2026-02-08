"""
Motor Anomaly Detection Web API
Serves your trained ML models as a web service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_predictor import MotorAnomalyPredictor
import pandas as pd
import logging
from datetime import datetime

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Initialize predictor globally
predictor = MotorAnomalyPredictor()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    """
    API home page with basic info
    """
    return {
        "message": "Motor Anomaly Detection API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/predict": "Single motor reading prediction",
            "/predict_batch": "Multiple motor readings prediction",
            "/health": "API health check",
            "/model_info": "Model information"
        },
        "usage": "Send POST requests with motor sensor data to get anomaly predictions"
    }

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    try:
        # Test if models are loaded
        info = predictor.get_model_info()
        if info:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": True,
                "best_model": info['best_model']
            }
        else:
            return {
                "status": "unhealthy", 
                "timestamp": datetime.now().isoformat(),
                "models_loaded": False,
                "error": "Models not loaded"
            }, 500
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(), 
            "error": str(e)
        }, 500

@app.route('/model_info')
def model_info():
    """
    Get information about the loaded models
    """
    try:
        info = predictor.get_model_info()
        if info:
            return {
                "best_model": info['best_model'],
                "performance": {
                    "random_forest_auc": info['rf_auc'],
                    "xgboost_auc": info['xgb_auc']
                },
                "training_samples": info['training_samples'],
                "features": info['feature_columns'],
                "top_features": info['feature_importance'][:3]
            }
        else:
            return {"error": "Model information not available"}, 500
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e)}, 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict anomaly for a single motor reading
    
    Expected JSON input:
    {
        "temperature": 75.5,
        "voltage": 24.2,
        "position": 180.0,
        "relative_time": 150.0,
        "temp_rolling_mean": 75.0,     // optional
        "voltage_rolling_std": 0.1,    // optional
        "motor_encoded": 0             // optional (0-5 for motors 1-6)
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return {"error": "No JSON data provided"}, 400
        
        # Validate required fields
        required_fields = ['temperature', 'voltage', 'position', 'relative_time']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }, 400
        
        # Make prediction
        result = predictor.predict_single(
            temperature=float(data['temperature']),
            voltage=float(data['voltage']),
            position=float(data['position']),
            relative_time=float(data['relative_time']),
            temp_rolling_mean=float(data.get('temp_rolling_mean', data['temperature'])),
            voltage_rolling_std=float(data.get('voltage_rolling_std', 0.1)),
            motor_encoded=int(data.get('motor_encoded', 0)),
            model_type=data.get('model_type', 'best')
        )
        
        if result:
            # Add timestamp to response
            result['timestamp'] = datetime.now().isoformat()
            result['api_version'] = "1.0"
            return result
        else:
            return {"error": "Prediction failed"}, 500
            
    except ValueError as e:
        return {"error": f"Invalid data type: {e}"}, 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}, 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict anomalies for multiple motor readings
    
    Expected JSON input:
    {
        "data": [
            {
                "temperature": 70.0,
                "voltage": 24.0,
                "position": 0,
                "relative_time": 10
            },
            {
                "temperature": 85.0,
                "voltage": 23.5,
                "position": 90,
                "relative_time": 30
            }
        ]
    }
    """
    try:
        # Get JSON data from request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return {"error": "No data array provided in JSON"}, 400
        
        data_list = request_data['data']
        
        if not isinstance(data_list, list) or len(data_list) == 0:
            return {"error": "Data must be a non-empty array"}, 400
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data_list)
        except Exception as e:
            return {"error": f"Could not create DataFrame from data: {e}"}, 400
        
        # Validate required columns
        required_cols = ['temperature', 'voltage', 'position', 'relative_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return {
                "error": f"Missing required columns: {missing_cols}",
                "required_columns": required_cols
            }, 400
        
        # Make batch prediction
        results_df = predictor.predict_batch(
            df, 
            model_type=request_data.get('model_type', 'best')
        )
        
        if results_df is not None:
            # Convert back to JSON-friendly format
            results = results_df.to_dict('records')
            
            # Add summary statistics
            total_samples = len(results)
            anomaly_count = sum(1 for r in results if r['is_anomaly'])
            
            return {
                "predictions": results,
                "summary": {
                    "total_samples": total_samples,
                    "anomalies_detected": anomaly_count,
                    "anomaly_rate": anomaly_count / total_samples if total_samples > 0 else 0,
                    "timestamp": datetime.now().isoformat(),
                    "api_version": "1.0"
                }
            }
        else:
            return {"error": "Batch prediction failed"}, 500
            
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return {"error": str(e)}, 500

@app.errorhandler(404)
def not_found(error):
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/predict", "/predict_batch", "/health", "/model_info"]
    }, 404

@app.errorhandler(500)
def internal_error(error):
    return {
        "error": "Internal server error",
        "message": "Something went wrong processing your request"
    }, 500

if __name__ == '__main__':
    print("Starting Motor Anomaly Detection API...")
    print("Loading ML models...")
    
    # Load models at startup
    if predictor.load_models():
        print("Models loaded successfully!")
        print("\nAPI Endpoints:")
        print("  GET  /           - API information")
        print("  GET  /health     - Health check")
        print("  GET  /model_info - Model information")
        print("  POST /predict    - Single prediction")
        print("  POST /predict_batch - Batch predictions")
        print("\nStarting server on http://localhost:5000")
        print("Press Ctrl+C to stop")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load models. Please run train_and_save_models.py first!")
        exit(1)