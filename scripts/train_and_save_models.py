"""
Simple model training and saving script
Creates models that others can use for predictions
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from data_loader import MotorDataLoader
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("TRAINING ML MODELS FOR MOTOR ANOMALY DETECTION")
    print("=" * 60)
    
    # Load data
    print("\nLoading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"   Loaded {len(anomaly_data)} samples")
    print(f"   Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Prepare features
    print("\nFeature Engineering...")
    feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
    
    # Add rolling features
    anomaly_data['temp_rolling_mean'] = anomaly_data.groupby(['session', 'motor_id'])['temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    anomaly_data['voltage_rolling_std'] = anomaly_data.groupby(['session', 'motor_id'])['voltage'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    
    # Add motor encoding
    motor_mapping = {f'motor_{i}': i-1 for i in range(1, 7)}
    anomaly_data['motor_encoded'] = anomaly_data['motor_id'].map(motor_mapping)
    
    feature_cols.extend(['temp_rolling_mean', 'voltage_rolling_std', 'motor_encoded'])
    
    print(f"   Created {len(feature_cols)} features")
    
    # Prepare data
    X = anomaly_data[feature_cols].fillna(0)
    y = anomaly_data['is_anomaly'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    rf_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    print(f"   Random Forest AUC: {rf_auc:.4f}")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    print(f"   XGBoost AUC: {xgb_auc:.4f}")
    
    # Model comparison
    print("\nMODEL COMPARISON")
    print("-" * 30)
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC:       {xgb_auc:.4f}")
    
    best_model = "Random Forest" if rf_auc > xgb_auc else "XGBoost"
    best_auc = max(rf_auc, xgb_auc)
    print(f"\nBest Model: {best_model} (AUC: {best_auc:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTOP FEATURES:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save models for others to use
    print(f"\nSaving trained models for reuse...")
    os.makedirs('saved_models', exist_ok=True)
    
    # Save models
    joblib.dump(rf, 'saved_models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'saved_models/xgboost_model.pkl')
    joblib.dump(scaler, 'saved_models/feature_scaler.pkl')
    
    # Save feature information
    model_info = {
        'feature_columns': feature_cols,
        'rf_auc': rf_auc,
        'xgb_auc': xgb_auc,
        'best_model': best_model,
        'training_samples': len(X_train),
        'feature_importance': feature_importance.to_dict('records')
    }
    joblib.dump(model_info, 'saved_models/model_info.pkl')
    
    print(f"Models saved to 'saved_models/' folder:")
    print(f"   Random Forest: random_forest_model.pkl")
    print(f"   XGBoost: xgboost_model.pkl") 
    print(f"   Feature Scaler: feature_scaler.pkl")
    print(f"   Model Info: model_info.pkl")
    print(f"\nOthers can now use your trained models!")
    
    print(f"\nSUCCESS! MODELS READY FOR USE")
    print("=" * 60)
    print(f"Best performing model: {best_model}")
    print(f"AUC Score: {best_auc:.4f}")
    
    if best_auc > 0.85:
        print("Excellent predictive performance!")
    elif best_auc > 0.75:
        print("Good predictive performance!")
    else:
        print("Moderate performance - consider more features")
    
    print("\nRun 'python use_trained_models.py' to see examples!")
    
    return {
        'rf_model': rf,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'rf_auc': rf_auc,
        'xgb_auc': xgb_auc,
        'best_model': best_model,
        'feature_importance': feature_importance,
        'feature_columns': feature_cols
    }

if __name__ == "__main__":
    results = main()