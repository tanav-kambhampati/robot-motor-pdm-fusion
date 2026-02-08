"""
🤖 MACHINE LEARNING TRAINING PIPELINE
Trains Random Forest, XGBoost, and LSTM models for motor predictive maintenance
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import MotorDataLoader
from ml_models import PredictiveMaintenanceModels
import warnings
warnings.filterwarnings('ignore')

def main():
    print("MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\nLoading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"   Loaded {len(anomaly_data)} samples")
    print(f"   Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Initialize ML models
    ml_models = PredictiveMaintenanceModels(anomaly_data)
    
    # Prepare features
    print("\nFeature Engineering...")
    feature_cols = ml_models.prepare_features()
    
    # Train models
    print(f"\nTraining {len(['Random Forest', 'XGBoost', 'LSTM'])} machine learning models...")
    
    # 1. Random Forest
    rf_model, rf_results = ml_models.train_random_forest()
    
    # 2. XGBoost  
    xgb_model, xgb_results = ml_models.train_xgboost()
    
    # 3. LSTM
    lstm_model, lstm_results = ml_models.train_lstm(sequence_length=30)
    
    # Model comparison
    print("\nComparing model performance...")
    comparison_df, best_model = ml_models.compare_models()
    
    # Create evaluation plots
    ml_models.create_model_evaluation_plots()
    
    # Feature importance analysis
    print("\nTOP FEATURE INSIGHTS:")
    print("-" * 40)
    
    if 'random_forest' in ml_models.results:
        rf_top_features = ml_models.results['random_forest']['feature_importance'].head(5)
        print("Random Forest Top Features:")
        for _, row in rf_top_features.iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    if 'xgboost' in ml_models.results:
        xgb_top_features = ml_models.results['xgboost']['feature_importance'].head(5)
        print("\nXGBoost Top Features:")
        for _, row in xgb_top_features.iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    # Model recommendations
    print(f"\nFINAL RECOMMENDATIONS:")
    print(f"   • Best Model: {best_model}")
    print(f"   • Best AUC Score: {comparison_df['AUC Score'].max():.4f}")
    
    if comparison_df['AUC Score'].max() > 0.85:
        print("   Excellent predictive performance achieved!")
    elif comparison_df['AUC Score'].max() > 0.75:
        print("   Good predictive performance achieved!")
    else:
        print("   Consider more feature engineering or data collection")
    
    # Deployment readiness
    print(f"\nDEPLOYMENT READINESS:")
    print(f"   • Models trained and evaluated")
    print(f"   • Feature pipeline established") 
    print(f"   • Anomaly detection calibrated")
    print(f"   • Evaluation plots generated")
    
    print(f"\nGenerated Files:")
    print(f"   • plots/model_evaluation.png")
    print(f"   • plots/lstm_training_history.png")
    
    return ml_models, comparison_df

if __name__ == "__main__":
    ml_models, comparison_df = main()