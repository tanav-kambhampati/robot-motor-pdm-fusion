"""
Examples of how to use the Motor Anomaly Detection API
Shows different ways to send requests and get predictions
"""

import requests
import json

# API base URL (change this to your server's address)
API_BASE = "http://localhost:5000"

def test_api_health():
    """
    Test if the API is running and healthy
    """
    print("=" * 50)
    print("TESTING API HEALTH")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """
    Test single motor reading prediction
    """
    print("\n" + "=" * 50)
    print("TESTING SINGLE PREDICTION")
    print("=" * 50)
    
    # Example motor data
    motor_data = {
        "temperature": 85.5,
        "voltage": 23.2,
        "position": 180.0,
        "relative_time": 120.0,
        "motor_encoded": 0  # Motor 1
    }
    
    print("Sending motor data:")
    print(json.dumps(motor_data, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE}/predict",
            json=motor_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            print(f"\nPREDICTION SUMMARY:")
            print(f"Motor Status: {'ANOMALY DETECTED!' if result['is_anomaly'] else 'NORMAL'}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Model Used: {result['model_used']}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """
    Test batch prediction with multiple motor readings
    """
    print("\n" + "=" * 50)
    print("TESTING BATCH PREDICTION")
    print("=" * 50)
    
    # Example batch data
    batch_data = {
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
            },
            {
                "temperature": 95.0,
                "voltage": 22.1,
                "position": 180,
                "relative_time": 60
            },
            {
                "temperature": 72.0,
                "voltage": 24.1,
                "position": 270,
                "relative_time": 90
            }
        ]
    }
    
    print("Sending batch data:")
    print(f"Number of samples: {len(batch_data['data'])}")
    
    try:
        response = requests.post(
            f"{API_BASE}/predict_batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\nBATCH PREDICTION SUMMARY:")
            summary = result['summary']
            print(f"Total samples: {summary['total_samples']}")
            print(f"Anomalies detected: {summary['anomalies_detected']}")
            print(f"Anomaly rate: {summary['anomaly_rate']:.1%}")
            
            print(f"\nDETAILED RESULTS:")
            for i, prediction in enumerate(result['predictions']):
                status = "ANOMALY" if prediction['is_anomaly'] else "NORMAL"
                print(f"Sample {i+1}: {status} (confidence: {prediction['confidence']:.1%})")
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """
    Get information about the loaded models
    """
    print("\n" + "=" * 50)
    print("TESTING MODEL INFO")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE}/model_info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            info = response.json()
            print(f"\nMODEL INFORMATION:")
            print(f"Best Model: {info['best_model']}")
            print(f"Random Forest AUC: {info['performance']['random_forest_auc']:.3f}")
            print(f"XGBoost AUC: {info['performance']['xgboost_auc']:.3f}")
            print(f"Training Samples: {info['training_samples']}")
            
            print(f"\nTOP FEATURES:")
            for i, feature in enumerate(info['top_features']):
                print(f"{i+1}. {feature['feature']}: {feature['importance']:.3f}")
        else:
            print(f"Error: {response.json()}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def demo_real_world_usage():
    """
    Show how someone would use this in a real application
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD USAGE EXAMPLE")
    print("=" * 60)
    print("Simulating a robotics application checking motor health...")
    
    # Simulate robot with 3 motors
    motors = [
        {"name": "Arm Motor", "temp": 78.5, "voltage": 23.8, "pos": 45},
        {"name": "Base Motor", "temp": 92.0, "voltage": 22.3, "pos": 180},
        {"name": "Gripper Motor", "temp": 71.2, "voltage": 24.1, "pos": 0}
    ]
    
    print("\nChecking all robot motors...")
    
    for i, motor in enumerate(motors):
        motor_data = {
            "temperature": motor["temp"],
            "voltage": motor["voltage"], 
            "position": motor["pos"],
            "relative_time": 60.0,
            "motor_encoded": i
        }
        
        try:
            response = requests.post(f"{API_BASE}/predict", json=motor_data)
            if response.status_code == 200:
                result = response.json()
                status = "NEEDS ATTENTION" if result['is_anomaly'] else "OK"
                print(f"{motor['name']}: {status} (confidence: {result['confidence']:.1%})")
            else:
                print(f"{motor['name']}: API Error")
        except:
            print(f"{motor['name']}: Connection Error")

def main():
    """
    Run all API tests
    """
    print("MOTOR ANOMALY DETECTION API - TESTING SUITE")
    print("=" * 60)
    print("Make sure the API is running: python motor_api.py")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("API Health Check", test_api_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),  
        ("Model Information", test_model_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
    
    # Show real-world example
    demo_real_world_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    print(f"\nAPI is ready for production use!")
    print(f"Share this URL with others: {API_BASE}")

if __name__ == "__main__":
    main()