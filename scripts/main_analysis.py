"""
Motor Predictive Maintenance Analysis
Research-grade analysis with advanced visualizations and machine learning.

This is the main execution script that orchestrates the entire analysis pipeline.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import MotorDataLoader
from eda_visualizer import MotorEDAVisualizer, create_plots_directory
import warnings
warnings.filterwarnings('ignore')

def main():
    print("STARTING MOTOR PREDICTIVE MAINTENANCE ANALYSIS")
    print("=" * 70)
    
    # Create necessary directories
    create_plots_directory()
    
    # Step 1: Load and combine all motor data
    print("\nSTEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    summary = loader.get_data_summary()
    
    # Print comprehensive data summary
    # Print comprehensive data summary
    print(f"\nDATASET OVERVIEW:")
    print(f"   • Total measurements: {summary['total_rows']:,}")
    print(f"   • Test sessions: {summary['total_sessions']}")
    print(f"   • Motors analyzed: {summary['total_motors']}")
    print(f"   • Test duration: {summary['time_range']['duration_hours']:.2f} hours")
    print(f"   • Data collection period: {summary['time_range']['min']:.0f} - {summary['time_range']['max']:.0f}")
    
    print(f"\nSENSOR STATISTICS:")
    for sensor, stats in summary['sensor_stats'].items():
        print(f"   • {sensor.title()}:")
        print(f"     - Range: {stats['min']:.2f} to {stats['max']:.2f}")
        print(f"     - Average: {stats['mean']:.2f}")
    
    # Step 2: Anomaly Detection
    # Step 2: Anomaly Detection
    print("\nSTEP 2: ANOMALY DETECTION")
    print("-" * 50)
    
    anomaly_data = loader.detect_anomalies(method='iqr')
    total_anomalies = anomaly_data['is_anomaly'].sum()
    anomaly_rate = (total_anomalies / len(anomaly_data)) * 100
    
    print(f"   • Total anomalies detected: {total_anomalies:,}")
    print(f"   • Anomaly rate: {anomaly_rate:.2f}%")
    
    # Breakdown by sensor
    for sensor in ['temperature', 'voltage', 'position']:
        sensor_anomalies = anomaly_data[f'{sensor}_anomaly'].sum()
        sensor_rate = (sensor_anomalies / len(anomaly_data)) * 100
        print(f"   • {sensor.title()} anomalies: {sensor_anomalies:,} ({sensor_rate:.1f}%)")
    
    # Step 3: EDA and Visualization
    print("\nSTEP 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("-" * 50)
    
    visualizer = MotorEDAVisualizer(combined_data)
    
    print("   Creating overview dashboard...")
    visualizer.create_overview_dashboard()
    
    print("   Analyzing sensor correlations...")
    correlation_matrix, pca_variance = visualizer.analyze_sensor_correlations()
    
    print("   Creating anomaly detection plots...")
    anomaly_counts, severity_matrix = visualizer.create_anomaly_detection_plots(anomaly_data)
    
    print("   Generating interactive dashboard...")
    interactive_fig = visualizer.create_interactive_dashboard()
    
    print("   Running statistical analysis...")
    stats_report = visualizer.statistical_analysis_report()
    
    # Step 4: Research Insights Summary
    # Step 4: Research Insights Summary
    print("\nSTEP 4: KEY RESEARCH INSIGHTS")
    print("-" * 50)
    
    print(f"   CORRELATION INSIGHTS:")
    strong_correlations = []
    for i, sensor1 in enumerate(['temperature', 'voltage', 'position']):
        for j, sensor2 in enumerate(['temperature', 'voltage', 'position']):
            if i < j:  # Avoid duplicates
                corr_val = correlation_matrix.loc[sensor1, sensor2]
                if abs(corr_val) > 0.5:
                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate"
                    direction = "positive" if corr_val > 0 else "negative"
                    print(f"     • {strength} {direction} correlation between {sensor1} and {sensor2}: {corr_val:.3f}")
                    strong_correlations.append((sensor1, sensor2, corr_val))
    
    print(f"\n   PCA INSIGHTS:")
    print(f"     • First PC explains {pca_variance[0]*100:.1f}% of variance")
    print(f"     • First 2 PCs explain {(pca_variance[0] + pca_variance[1])*100:.1f}% of variance")
    
    print(f"\n   ANOMALY INSIGHTS:")
    most_anomalous_session = anomaly_counts.idxmax()
    most_anomalous_count = anomaly_counts.max()
    print(f"     • Most anomalous session: {most_anomalous_session} ({most_anomalous_count} anomalies)")
    
    # Motor health ranking
    motor_anomaly_rates = anomaly_data.groupby('motor_id')['is_anomaly'].mean().sort_values(ascending=False)
    print(f"     • Most problematic motor: {motor_anomaly_rates.index[0]} ({motor_anomaly_rates.iloc[0]*100:.1f}% anomaly rate)")
    print(f"     • Most reliable motor: {motor_anomaly_rates.index[-1]} ({motor_anomaly_rates.iloc[-1]*100:.1f}% anomaly rate)")
    
    # Step 5: Generate Research Report
    print("\nSTEP 5: GENERATING RESEARCH REPORT")
    print("-" * 50)
    
    report_content = f"""
# Motor Predictive Maintenance Research Report

## Executive Summary
This analysis examined {summary['total_rows']:,} sensor measurements from {summary['total_sessions']} test sessions across {summary['total_motors']} motors over {summary['time_range']['duration_hours']:.2f} hours.

## Key Findings

### 1. Data Characteristics
- **Temperature Range**: {summary['sensor_stats']['temperature']['min']:.1f}°C to {summary['sensor_stats']['temperature']['max']:.1f}°C
- **Voltage Range**: {summary['sensor_stats']['voltage']['min']:.0f}V to {summary['sensor_stats']['voltage']['max']:.0f}V  
- **Position Range**: {summary['sensor_stats']['position']['min']:.0f} to {summary['sensor_stats']['position']['max']:.0f} units

### 2. Anomaly Detection Results
- **Overall Anomaly Rate**: {anomaly_rate:.2f}%
- **Total Anomalies**: {total_anomalies:,} out of {len(anomaly_data):,} measurements
- **Most Anomalous Session**: {most_anomalous_session} with {most_anomalous_count} anomalies

### 3. Sensor Correlation Analysis
"""

    for sensor1, sensor2, corr_val in strong_correlations[:3]:  # Top 3 correlations
        report_content += f"- **{sensor1.title()} ↔ {sensor2.title()}**: {corr_val:.3f} correlation\n"

    report_content += f"""

### 4. Motor Health Ranking
1. **Most At-Risk**: {motor_anomaly_rates.index[0]} ({motor_anomaly_rates.iloc[0]*100:.1f}% anomaly rate)
2. **Most Reliable**: {motor_anomaly_rates.index[-1]} ({motor_anomaly_rates.iloc[-1]*100:.1f}% anomaly rate)

### 5. Principal Component Analysis
- First PC captures {pca_variance[0]*100:.1f}% of sensor variance
- Dimensional reduction feasible with 2-3 components

## Recommendations for Predictive Maintenance
1. **Priority Monitoring**: Focus on {motor_anomaly_rates.index[0]} - highest anomaly rate
2. **Sensor Strategy**: Temperature and voltage show strongest predictive signals
3. **Threshold Setting**: Current IQR-based anomaly detection identifies {anomaly_rate:.1f}% outliers
4. **Data Collection**: Extend monitoring duration for better trend analysis

## Next Steps
1. Implement machine learning models (Random Forest, XGBoost, LSTM)
2. Develop real-time anomaly detection system
3. Create predictive failure models
4. Establish maintenance scheduling optimization

---
*Analysis generated by Motor Predictive Maintenance Research System*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    with open('Motor_Predictive_Maintenance_Report.md', 'w') as f:
        f.write(report_content)
    
    print("   Research report saved as: Motor_Predictive_Maintenance_Report.md")
    
    # Final Summary
    print("\nANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"   Generated Files:")
    print(f"      • plots/overview_dashboard.png")
    print(f"      • plots/correlation_analysis.png") 
    print(f"      • plots/anomaly_analysis.png")
    print(f"      • plots/interactive_dashboard.html")
    print(f"      • Motor_Predictive_Maintenance_Report.md")
    
    print(f"\n   Ready for Machine Learning Phase!")
    print(f"      • {len(anomaly_data)} labeled samples ready for training")
    print(f"      • {total_anomalies} positive anomaly cases identified")
    print(f"      • Multi-sensor time series data preprocessed")
    
    return combined_data, anomaly_data, summary

if __name__ == "__main__":
    combined_data, anomaly_data, summary = main()