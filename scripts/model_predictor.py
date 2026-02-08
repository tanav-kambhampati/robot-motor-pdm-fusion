"""
ML Model Predictor Interface
Allows others to use your trained models for predictions.
"""

import joblib
import pandas as pd
import numpy as np
import os

class MotorAnomalyPredictor:
    """
    Easy-to-use interface for predicting motor anomalies
    """
    
    def __init__(self, models_dir='saved_models'):
        """
        Initialize the predictor by loading saved models
        
        Args:
            models_dir: Directory containing saved model files
        """
        self.models_dir = models_dir
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        self.model_info = None
        self.is_loaded = False
        
    def load_models(self):
        """
        Load all saved models and preprocessing components
        """
        try:
            print("Loading trained models...")
            
            # Load models
            self.rf_model = joblib.load(os.path.join(self.models_dir, 'random_forest_model.pkl'))
            self.xgb_model = joblib.load(os.path.join(self.models_dir, 'xgboost_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.models_dir, 'feature_scaler.pkl'))
            self.model_info = joblib.load(os.path.join(self.models_dir, 'model_info.pkl'))
            
            self.is_loaded = True
            print("Models loaded successfully!")
            print(f"   Best model: {self.model_info['best_model']}")
            print(f"   Training AUC: {self.model_info['rf_auc']:.3f} (RF), {self.model_info['xgb_auc']:.3f} (XGB)")
            print(f"   Trained on {self.model_info['training_samples']} samples")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found in '{self.models_dir}'")
            print("   Run the training script first to generate models!")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_single(self, temperature, voltage, position, relative_time, 
                      temp_rolling_mean=None, voltage_rolling_std=None, 
                      motor_encoded=0, model_type='best'):
        """
        Predict anomaly for a single motor reading
        
        Args:
            temperature: Motor temperature
            voltage: Motor voltage  
            position: Motor position
            relative_time: Time since start
            temp_rolling_mean: Optional rolling average of temperature
            voltage_rolling_std: Optional rolling std of voltage
            motor_encoded: Motor ID encoded (0-5 for motors 1-6)
            model_type: 'rf', 'xgb', or 'best' (default)
        
        Returns:
            dict with prediction results
        """
        if not self.is_loaded:
            if not self.load_models():
                return None
        
        # Use current values as rolling features if not provided
        if temp_rolling_mean is None:
            temp_rolling_mean = temperature
        if voltage_rolling_std is None:
            voltage_rolling_std = 0.1  # Small default std
            
        # Create feature array in the correct order
        features = np.array([[temperature, voltage, position, relative_time, 
                            temp_rolling_mean, voltage_rolling_std, motor_encoded]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Choose model
        if model_type == 'best':
            model = self.rf_model if self.model_info['best_model'] == 'Random Forest' else self.xgb_model
            model_name = self.model_info['best_model']
        elif model_type == 'rf':
            model = self.rf_model
            model_name = 'Random Forest'
        elif model_type == 'xgb':
            model = self.xgb_model
            model_name = 'XGBoost'
        else:
            raise ValueError("model_type must be 'rf', 'xgb', or 'best'")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            'is_anomaly': bool(prediction),
            'anomaly_probability': float(probability[1]),
            'normal_probability': float(probability[0]),
            'confidence': float(max(probability)),
            'model_used': model_name,
            'input_features': {
                'temperature': temperature,
                'voltage': voltage, 
                'position': position,
                'relative_time': relative_time,
                'temp_rolling_mean': temp_rolling_mean,
                'voltage_rolling_std': voltage_rolling_std,
                'motor_encoded': motor_encoded
            }
        }
        
        return result
    
    def predict_batch(self, data_df, model_type='best'):
        """
        Predict anomalies for multiple motor readings
        
        Args:
            data_df: DataFrame with columns ['temperature', 'voltage', 'position', 'relative_time']
                    Optional: ['temp_rolling_mean', 'voltage_rolling_std', 'motor_encoded']
            model_type: 'rf', 'xgb', or 'best'
        
        Returns:
            DataFrame with predictions added
        """
        if not self.is_loaded:
            if not self.load_models():
                return None
        
        df = data_df.copy()
        
        # Add rolling features if missing
        if 'temp_rolling_mean' not in df.columns:
            df['temp_rolling_mean'] = df['temperature']
        if 'voltage_rolling_std' not in df.columns:
            df['voltage_rolling_std'] = 0.1
        if 'motor_encoded' not in df.columns:
            df['motor_encoded'] = 0  # Default to motor 1
        
        # Prepare features
        feature_cols = ['temperature', 'voltage', 'position', 'relative_time', 
                       'temp_rolling_mean', 'voltage_rolling_std', 'motor_encoded']
        X = df[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Choose model
        if model_type == 'best':
            model = self.rf_model if self.model_info['best_model'] == 'Random Forest' else self.xgb_model
        elif model_type == 'rf':
            model = self.rf_model
        elif model_type == 'xgb':
            model = self.xgb_model
        else:
            raise ValueError("model_type must be 'rf', 'xgb', or 'best'")
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Add results to dataframe
        df['is_anomaly'] = predictions.astype(bool)
        df['anomaly_probability'] = probabilities[:, 1]
        df['normal_probability'] = probabilities[:, 0]
        df['confidence'] = np.max(probabilities, axis=1)
        
        return df
    
    def get_model_info(self):
        """
        Get information about the loaded models
        """
        if not self.is_loaded:
            if not self.load_models():
                return None
                
        return self.model_info
    
    def get_feature_importance(self):
        """
        Get feature importance from the Random Forest model
        """
        if not self.is_loaded:
            if not self.load_models():
                return None
        
        return pd.DataFrame(self.model_info['feature_importance'])


def demo_usage():
    """
    Demonstrate how to use the predictor
    """
    print("MOTOR ANOMALY PREDICTOR DEMO")
    print("=" * 50)
    
    # Create predictor
    predictor = MotorAnomalyPredictor()
    
    # Example 1: Single prediction
    print("\n1. Single Prediction Example:")
    result = predictor.predict_single(
        temperature=75.5,
        voltage=24.2, 
        position=180.0,
        relative_time=150.0
    )
    
    if result:
        print(f"   Input: Temp={result['input_features']['temperature']}C, "
              f"Voltage={result['input_features']['voltage']}V")
        print(f"   Prediction: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Model: {result['model_used']}")
    
    # Example 2: Batch prediction
    print("\n2. Batch Prediction Example:")
    sample_data = pd.DataFrame({
        'temperature': [70.0, 85.0, 95.0, 72.0],
        'voltage': [24.0, 23.5, 22.8, 24.1],
        'position': [0, 90, 180, 270],
        'relative_time': [10, 50, 100, 150]
    })
    
    results_df = predictor.predict_batch(sample_data)
    if results_df is not None:
        print("   Results:")
        for i, row in results_df.iterrows():
            status = "ANOMALY" if row['is_anomaly'] else "NORMAL"
            print(f"      Sample {i+1}: {status} (confidence: {row['confidence']:.1%})")
    
    # Example 3: Model information
    print("\n3. Model Information:")
    info = predictor.get_model_info()
    if info:
        print(f"   Best model: {info['best_model']}")
        print(f"   Performance: RF={info['rf_auc']:.3f}, XGB={info['xgb_auc']:.3f}")
        
    print("\nDemo complete! Your models are ready to use!")


if __name__ == "__main__":
    demo_usage()