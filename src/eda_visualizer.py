"""
Comprehensive EDA and Visualization Suite for Motor Predictive Maintenance
Research-grade plots and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MotorEDAVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sensor_cols = ['temperature', 'voltage', 'position']
        
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔧 Motor Sensor Data - Research Overview Dashboard', fontsize=16, y=0.98)
        
        # 1. Time series plot for each sensor
        for i, sensor in enumerate(self.sensor_cols):
            ax = axes[0, i]
            
            # Plot sample from each session
            for session in self.data['session'].unique()[:3]:  # Show first 3 sessions
                session_data = self.data[self.data['session'] == session].iloc[::50]  # Sample every 50th point
                ax.plot(session_data['relative_time'], session_data[sensor], 
                       alpha=0.7, label=f'Session {session[-6:]}')
            
            ax.set_title(f'{sensor.title()} Over Time')
            ax.set_xlabel('Relative Time (s)')
            ax.set_ylabel(sensor.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Distribution plots
        for i, sensor in enumerate(self.sensor_cols):
            ax = axes[1, i]
            
            # Create histogram with KDE
            self.data[sensor].hist(bins=50, alpha=0.7, ax=ax, density=True)
            
            # Add KDE curve
            x_range = np.linspace(self.data[sensor].min(), self.data[sensor].max(), 100)
            kde = stats.gaussian_kde(self.data[sensor].dropna())
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_title(f'{sensor.title()} Distribution')
            ax.set_xlabel(sensor.title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_sensor_correlations(self):
        """Deep dive into sensor correlations and relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔍 Sensor Correlation Analysis', fontsize=16)
        
        # 1. Correlation heatmap
        ax1 = axes[0, 0]
        correlation_matrix = self.data[self.sensor_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Sensor Correlation Matrix')
        
        # 2. Pairplot for sensor relationships
        ax2 = axes[0, 1]
        # Sample data for performance
        sample_data = self.data.sample(n=min(5000, len(self.data)))
        ax2.scatter(sample_data['temperature'], sample_data['voltage'], 
                   alpha=0.5, c=sample_data['position'], cmap='viridis')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Voltage')
        ax2.set_title('Temperature vs Voltage (colored by Position)')
        
        # 3. Statistical tests for each pair
        ax3 = axes[1, 0]
        correlations = []
        p_values = []
        pairs = []
        
        from itertools import combinations
        for sensor1, sensor2 in combinations(self.sensor_cols, 2):
            corr, p_val = stats.pearsonr(self.data[sensor1].dropna(), 
                                       self.data[sensor2].dropna())
            correlations.append(corr)
            p_values.append(p_val)
            pairs.append(f'{sensor1}-{sensor2}')
        
        bars = ax3.bar(pairs, correlations, color=['red' if p > 0.05 else 'green' for p in p_values])
        ax3.set_title('Pearson Correlations (Green=Significant)')
        ax3.set_ylabel('Correlation Coefficient')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add significance annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'p={p_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. PCA Analysis
        ax4 = axes[1, 1]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[self.sensor_cols].dropna())
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        ax4.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, label='Individual')
        ax4.plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-', label='Cumulative')
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title('PCA - Explained Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix, explained_var
    
    def create_anomaly_detection_plots(self, anomaly_data: pd.DataFrame):
        """Visualize anomaly detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚨 Anomaly Detection Analysis', fontsize=16)
        
        # 1. Anomaly timeline
        ax1 = axes[0, 0]
        anomaly_counts = anomaly_data.groupby('session')['is_anomaly'].sum()
        ax1.bar(range(len(anomaly_counts)), anomaly_counts.values)
        ax1.set_title('Anomalies by Session')
        ax1.set_xlabel('Session')
        ax1.set_ylabel('Number of Anomalies')
        ax1.set_xticks(range(len(anomaly_counts)))
        ax1.set_xticklabels([s[-6:] for s in anomaly_counts.index], rotation=45)
        
        # 2. Sensor-wise anomaly distribution
        ax2 = axes[0, 1]
        anomaly_by_sensor = []
        for sensor in self.sensor_cols:
            anomaly_by_sensor.append(anomaly_data[f'{sensor}_anomaly'].sum())
        
        ax2.pie(anomaly_by_sensor, labels=self.sensor_cols, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Anomalies by Sensor Type')
        
        # 3. 3D scatter plot of anomalies
        ax3 = axes[1, 0]
        normal_data = anomaly_data[~anomaly_data['is_anomaly']].sample(n=min(1000, len(anomaly_data)))
        anomaly_points = anomaly_data[anomaly_data['is_anomaly']]
        
        ax3.scatter(normal_data['temperature'], normal_data['voltage'], 
                   alpha=0.5, c='blue', label='Normal', s=10)
        ax3.scatter(anomaly_points['temperature'], anomaly_points['voltage'], 
                   alpha=0.8, c='red', label='Anomaly', s=20, marker='x')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Voltage')
        ax3.set_title('Anomalies in Temperature-Voltage Space')
        ax3.legend()
        
        # 4. Anomaly severity heatmap
        ax4 = axes[1, 1]
        # Create severity matrix (motor vs session)
        severity_matrix = anomaly_data.pivot_table(
            values='is_anomaly', 
            index='motor_id', 
            columns='session', 
            aggfunc='mean'
        ).fillna(0)
        
        sns.heatmap(severity_matrix, annot=True, cmap='Reds', ax=ax4, cbar_kws={'label': 'Anomaly Rate'})
        ax4.set_title('Anomaly Rate by Motor and Session')
        ax4.set_xlabel('Session')
        ax4.set_ylabel('Motor ID')
        
        plt.tight_layout()
        plt.savefig('plots/anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return anomaly_counts, severity_matrix
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Sample data for performance
        sample_data = self.data.sample(n=min(10000, len(self.data)))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sensor Readings Over Time', 'Temperature vs Voltage', 
                          'Sensor Distributions', 'Session Comparison'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "box"}]]
        )
        
        # 1. Time series with multiple sensors
        colors = ['blue', 'red', 'green']
        for i, sensor in enumerate(self.sensor_cols):
            fig.add_trace(
                go.Scatter(x=sample_data['relative_time'], y=sample_data[sensor],
                          mode='lines', name=sensor.title(), opacity=0.7,
                          line=dict(color=colors[i])),
                row=1, col=1
            )
        
        # 2. Scatter plot with color coding
        fig.add_trace(
            go.Scatter(x=sample_data['temperature'], y=sample_data['voltage'],
                      mode='markers', name='Measurements',
                      marker=dict(color=sample_data['position'], colorscale='Viridis',
                                size=5, opacity=0.6, colorbar=dict(title="Position"))),
            row=1, col=2
        )
        
        # 3. Histograms for each sensor
        for sensor in self.sensor_cols:
            fig.add_trace(
                go.Histogram(x=sample_data[sensor], name=f'{sensor.title()} Dist',
                           opacity=0.7, nbinsx=30),
                row=2, col=1
            )
        
        # 4. Box plots by session
        for sensor in self.sensor_cols:
            fig.add_trace(
                go.Box(y=sample_data[sensor], x=sample_data['session'],
                      name=f'{sensor.title()}', boxpoints='outliers'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="🔧 Interactive Motor Sensor Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800
        )
        
        fig.write_html("plots/interactive_dashboard.html")
        print("📊 Interactive dashboard saved to plots/interactive_dashboard.html")
        return fig
    
    def statistical_analysis_report(self):
        """Generate comprehensive statistical analysis"""
        print("📈 COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("=" * 60)
        
        # Basic statistics
        print("\n1. DESCRIPTIVE STATISTICS:")
        print(self.data[self.sensor_cols].describe())
        
        # Normality tests
        print("\n2. NORMALITY TESTS (Shapiro-Wilk):")
        for sensor in self.sensor_cols:
            # Sample for performance (Shapiro-Wilk has limitations on large datasets)
            sample = self.data[sensor].dropna().sample(n=min(5000, len(self.data)))
            statistic, p_value = stats.shapiro(sample)
            is_normal = "✅ Normal" if p_value > 0.05 else "❌ Non-normal"
            print(f"  {sensor.title()}: {is_normal} (p={p_value:.2e})")
        
        # Stationarity test (for time series)
        print("\n3. STATIONARITY TESTS (Augmented Dickey-Fuller):")
        from statsmodels.tts.stattools import adfuller
        
        for sensor in self.sensor_cols:
            result = adfuller(self.data[sensor].dropna())
            is_stationary = "✅ Stationary" if result[1] <= 0.05 else "❌ Non-stationary"
            print(f"  {sensor.title()}: {is_stationary} (p={result[1]:.2e})")
        
        # Variance analysis
        print("\n4. VARIANCE ANALYSIS BY SESSION:")
        variance_analysis = self.data.groupby('session')[self.sensor_cols].var()
        print(variance_analysis)
        
        # Motor-specific analysis
        print("\n5. MOTOR PERFORMANCE COMPARISON:")
        motor_stats = self.data.groupby('motor_id')[self.sensor_cols].agg(['mean', 'std'])
        print(motor_stats)
        
        return {
            'descriptive_stats': self.data[self.sensor_cols].describe(),
            'variance_by_session': variance_analysis,
            'motor_performance': motor_stats
        }

def create_plots_directory():
    """Ensure plots directory exists"""
    import os
    os.makedirs('plots', exist_ok=True)

if __name__ == "__main__":
    # This will be called from main analysis script
    pass