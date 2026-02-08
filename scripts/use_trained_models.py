"""
🎯 How to Use the Trained ML Models
Simple examples for anyone to use your trained models!
"""

from model_predictor import MotorAnomalyPredictor
import pandas as pd

def main():
    print("🚀 USING TRAINED ML MODELS - EXAMPLES")
    print("=" * 60)
    print("This shows how anyone can use your trained models!")
    print()
    
    # Create the predictor
    predictor = MotorAnomalyPredictor()
    
    print("📖 EXAMPLE 1: Check if a single motor reading is normal")
    print("-" * 50)
    
    # Example of a normal reading
    result1 = predictor.predict_single(
        temperature=72.0,    # Normal temperature
        voltage=24.0,        # Good voltage
        position=45.0,       # Position angle
        relative_time=30.0   # 30 seconds into operation
    )
    
    if result1:
        print(f"🌡️  Motor Status Check:")
        print(f"   Temperature: {result1['input_features']['temperature']}°C")
        print(f"   Voltage: {result1['input_features']['voltage']}V")
        print(f"   Result: {'🚨 POTENTIAL ISSUE!' if result1['is_anomaly'] else '✅ LOOKS NORMAL'}")
        print(f"   Confidence: {result1['confidence']:.1%}")
        print()
    
    print("📖 EXAMPLE 2: Check a suspicious reading")
    print("-" * 50)
    
    # Example of potentially problematic reading
    result2 = predictor.predict_single(
        temperature=95.0,    # High temperature!
        voltage=22.5,        # Low voltage
        position=180.0,      
        relative_time=200.0  
    )
    
    if result2:
        print(f"🌡️  Motor Status Check:")
        print(f"   Temperature: {result2['input_features']['temperature']}°C")
        print(f"   Voltage: {result2['input_features']['voltage']}V") 
        print(f"   Result: {'🚨 POTENTIAL ISSUE!' if result2['is_anomaly'] else '✅ LOOKS NORMAL'}")
        print(f"   Confidence: {result2['confidence']:.1%}")
        print()
    
    print("📖 EXAMPLE 3: Analyze multiple motor readings at once")
    print("-" * 50)
    
    # Create sample data that someone might have
    motor_data = pd.DataFrame({
        'temperature': [70.0, 75.0, 88.0, 95.0, 72.0, 91.0],
        'voltage': [24.0, 23.8, 23.2, 22.1, 24.1, 22.8],
        'position': [0, 60, 120, 180, 240, 300],
        'relative_time': [10, 30, 60, 90, 120, 150]
    })
    
    print("🔍 Input Data:")
    print(motor_data.to_string(index=False))
    print()
    
    # Get predictions for all readings
    results = predictor.predict_batch(motor_data)
    
    if results is not None:
        print("📊 Analysis Results:")
        print("-" * 30)
        for i, row in results.iterrows():
            status_emoji = "🚨" if row['is_anomaly'] else "✅"
            status_text = "ISSUE" if row['is_anomaly'] else "NORMAL"
            print(f"   Reading {i+1}: {status_emoji} {status_text} "
                  f"(Temp: {row['temperature']}°C, Confidence: {row['confidence']:.1%})")
        
        anomaly_count = results['is_anomaly'].sum()
        print(f"\n📈 Summary: {anomaly_count}/{len(results)} readings flagged as potential issues")
        print()
    
    print("📖 EXAMPLE 4: Get model performance info")
    print("-" * 50)
    
    info = predictor.get_model_info()
    if info:
        print(f"🏆 Best performing model: {info['best_model']}")
        print(f"📊 Model accuracy scores:")
        print(f"   Random Forest: {info['rf_auc']:.3f}")
        print(f"   XGBoost: {info['xgb_auc']:.3f}")
        print(f"📈 Trained on {info['training_samples']} motor samples")
        
        print(f"\n🎯 Most important features for prediction:")
        importance_df = pd.DataFrame(info['feature_importance'])
        top_features = importance_df.head(3)
        for _, feature in top_features.iterrows():
            print(f"   {feature['feature']}: {feature['importance']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! You now know how to use the trained models!")
    print()
    print("💡 What you can do:")
    print("   ✅ Check single motor readings for problems")
    print("   ✅ Analyze batches of data from CSV files") 
    print("   ✅ Build this into monitoring systems")
    print("   ✅ Use in robotics projects for predictive maintenance")
    print()
    print("🚀 This is how real AI gets deployed in robotics!")
    print("=" * 60)

if __name__ == "__main__":
    main()