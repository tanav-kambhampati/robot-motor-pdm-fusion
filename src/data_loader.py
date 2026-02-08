"""
Motor Sensor Data Loader
Handles loading and initial processing of sensor data from multiple test sessions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MotorDataLoader:
    def __init__(self, data_path: str = "data/raw/testing_data"):
        self.data_path = Path(data_path)
        self.raw_data = {}
        self.combined_data = None
        
    def load_all_sessions(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all motor data from all test sessions"""
        # print("Loading motor sensor data from all test sessions...")
        
        session_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        for session_dir in sorted(session_dirs):
            session_name = session_dir.name
            self.raw_data[session_name] = {}
            
            # Load all motor files in this session
            csv_files = list(session_dir.glob("data_motor_*.csv"))
            
            for csv_file in sorted(csv_files):
                motor_name = csv_file.stem  # e.g., "data_motor_1"
                
                try:
                    df = pd.read_csv(csv_file)
                    # Add metadata columns
                    df['session'] = session_name
                    df['motor_id'] = motor_name
                    df['file_path'] = str(csv_file)
                    
                    self.raw_data[session_name][motor_name] = df
                    # print(f"Loaded {motor_name} from {session_name}: {len(df)} rows")
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
        
        return self.raw_data
    
    def combine_all_data(self) -> pd.DataFrame:
        """Combine all motor data into single DataFrame"""
        if not self.raw_data:
            self.load_all_sessions()
        
        all_dataframes = []
        
        for session_name, motors in self.raw_data.items():
            for motor_name, df in motors.items():
                all_dataframes.append(df)
        
        self.combined_data = pd.concat(all_dataframes, ignore_index=True)
        
        # Convert time to relative time within each session
        self.combined_data['relative_time'] = self.combined_data.groupby(['session', 'motor_id'])['time'].transform(
            lambda x: x - x.min()
        )
        
        print(f"Combined dataset: {len(self.combined_data)} rows across {len(self.raw_data)} sessions")
        return self.combined_data
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of the dataset"""
        if self.combined_data is None:
            self.combine_all_data()
        
        summary = {
            'total_rows': len(self.combined_data),
            'total_sessions': self.combined_data['session'].nunique(),
            'total_motors': self.combined_data['motor_id'].nunique(),
            'time_range': {
                'min': self.combined_data['time'].min(),
                'max': self.combined_data['time'].max(),
                'duration_hours': (self.combined_data['time'].max() - self.combined_data['time'].min()) / 3600
            },
            'sensor_stats': {
                'temperature': {
                    'min': self.combined_data['temperature'].min(),
                    'max': self.combined_data['temperature'].max(),
                    'mean': self.combined_data['temperature'].mean()
                },
                'voltage': {
                    'min': self.combined_data['voltage'].min(),
                    'max': self.combined_data['voltage'].max(),
                    'mean': self.combined_data['voltage'].mean()
                },
                'position': {
                    'min': self.combined_data['position'].min(),
                    'max': self.combined_data['position'].max(),
                    'mean': self.combined_data['position'].mean()
                }
            },
            'missing_values': self.combined_data.isnull().sum().to_dict()
        }
        
        return summary
    
    def detect_anomalies(self, method='iqr', threshold=3.0) -> pd.DataFrame:
        """Detect anomalies in sensor readings"""
        if self.combined_data is None:
            self.combine_all_data()
        
        df = self.combined_data.copy()
        sensor_cols = ['temperature', 'voltage', 'position']
        
        for col in sensor_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[f'{col}_anomaly'] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df[f'{col}_anomaly'] = z_scores > threshold
        
        # Create combined anomaly flag
        anomaly_cols = [f'{col}_anomaly' for col in sensor_cols]
        df['is_anomaly'] = df[anomaly_cols].any(axis=1)
        
        return df

if __name__ == "__main__":
    # Test the data loader
    loader = MotorDataLoader()
    data = loader.combine_all_data()
    summary = loader.get_data_summary()
    
    print("\nDATA SUMMARY:")
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Sessions: {summary['total_sessions']}")
    print(f"Motors: {summary['total_motors']}")
    print(f"Duration: {summary['time_range']['duration_hours']:.2f} hours")