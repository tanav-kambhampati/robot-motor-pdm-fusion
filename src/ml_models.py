"""
Machine Learning Models for Motor Predictive Maintenance
Implements Random Forest, XGBoost, and LSTM models with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceModels:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_features(self, sequence_length=50):
        """Prepare features for ML models"""
        print("Preparing features for machine learning...")
        
        # Basic features
        feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
        
        # Engineer additional features
        self.data['temp_rolling_mean'] = self.data.groupby(['session', 'motor_id'])['temperature'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        self.data['voltage_rolling_std'] = self.data.groupby(['session', 'motor_id'])['voltage'].transform(
            lambda x: x.rolling(window=10, min_periods=1).std()
        )
        self.data['position_diff'] = self.data.groupby(['session', 'motor_id'])['position'].diff().fillna(0)
        
        # Temperature-voltage interaction
        self.data['temp_voltage_interaction'] = self.data['temperature'] * self.data['voltage']
        
        # Trend features
        self.data['temp_trend'] = self.data.groupby(['session', 'motor_id'])['temperature'].transform(
            lambda x: x.diff().fillna(0)
        )
        self.data['voltage_trend'] = self.data.groupby(['session', 'motor_id'])['voltage'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        feature_cols.extend([
            'temp_rolling_mean', 'voltage_rolling_std', 'position_diff',
            'temp_voltage_interaction', 'temp_trend', 'voltage_trend'
        ])
        
        # Encode categorical variables
        le_session = LabelEncoder()
        le_motor = LabelEncoder()
        
        self.data['session_encoded'] = le_session.fit_transform(self.data['session'])
        self.data['motor_encoded'] = le_motor.fit_transform(self.data['motor_id'])
        
        feature_cols.extend(['session_encoded', 'motor_encoded'])
        
        self.feature_columns = feature_cols
        self.feature_columns = feature_cols
        print(f"   Created {len(feature_cols)} features for modeling")
        
        return feature_cols
    
    def train_random_forest(self, test_size=0.2, random_state=42):
        """Train Random Forest model with hyperparameter tuning"""
        print("\nTraining Random Forest Model...")
        
        # Prepare data
        X = self.data[self.feature_columns].fillna(0)
        y = self.data['is_anomaly'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['random_forest'] = scaler
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=random_state)
        
        print("   Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        self.models['random_forest'] = best_rf
        
        # Predictions
        y_pred = best_rf.predict(X_test_scaled)
        y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        self.results['random_forest'] = {
            'model': best_rf,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_,
            'feature_importance': pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_rf.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        print(f"   Random Forest AUC: {auc_score:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        
        return best_rf, self.results['random_forest']
    
    def train_xgboost(self, test_size=0.2, random_state=42):
        """Train XGBoost model with hyperparameter tuning"""
        print("\nTraining XGBoost Model...")
        
        # Prepare data
        X = self.data[self.feature_columns].fillna(0)
        y = self.data['is_anomaly'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc'
        )
        
        print("   Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        self.models['xgboost'] = best_xgb
        
        # Predictions
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        # Evaluation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='roc_auc')
        
        self.results['xgboost'] = {
            'model': best_xgb,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_,
            'feature_importance': pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_xgb.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        print(f"   XGBoost AUC: {auc_score:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        
        return best_xgb, self.results['xgboost']
    
    def prepare_lstm_sequences(self, sequence_length=50, test_size=0.2, random_state=42):
        """Prepare sequences for LSTM model"""
        print(f"\nPreparing LSTM sequences (length={sequence_length})...")
        
        # Sort data by session, motor, and time
        sorted_data = self.data.sort_values(['session', 'motor_id', 'relative_time']).reset_index(drop=True)
        
        # Use only numerical sensor features for LSTM
        sensor_features = ['temperature', 'voltage', 'position']
        
        sequences = []
        labels = []
        
        # Create sequences for each motor in each session
        for session in sorted_data['session'].unique():
            for motor in sorted_data['motor_id'].unique():
                motor_data = sorted_data[
                    (sorted_data['session'] == session) & 
                    (sorted_data['motor_id'] == motor)
                ][sensor_features + ['is_anomaly']].values
                
                if len(motor_data) < sequence_length:
                    continue
                
                # Create sliding window sequences
                for i in range(len(motor_data) - sequence_length + 1):
                    seq = motor_data[i:i+sequence_length, :-1]  # Features
                    label = motor_data[i+sequence_length-1, -1]  # Last label in sequence
                    sequences.append(seq)
                    labels.append(label)
        
        X_lstm = np.array(sequences)
        y_lstm = np.array(labels)
        
        print(f"   Created {len(sequences)} sequences of shape {X_lstm.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_lstm, y_lstm, test_size=test_size, random_state=random_state, stratify=y_lstm
        )
        
        # Scale features
        scaler = StandardScaler()
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)
        
        # Scale train and test
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        self.scalers['lstm'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_lstm(self, sequence_length=50, test_size=0.2, random_state=42):
        """Train LSTM model for time series prediction"""
        print("\nTraining LSTM Model...")
        
        # Prepare sequences
        X_train, X_test, y_train, y_test = self.prepare_lstm_sequences(
            sequence_length, test_size, random_state
        )
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, 3)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print("   Training LSTM...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        self.models['lstm'] = model
        self.results['lstm'] = {
            'model': model,
            'history': history,
            'auc_score': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'sequence_length': sequence_length
        }
        
        print(f"   LSTM AUC: {auc_score:.4f}")
        print(f"   Training completed in {len(history.history['loss'])} epochs")
        
        return model, self.results['lstm']
    
    def compare_models(self):
        """Compare all trained models"""
        print("\nMODEL COMPARISON RESULTS")
        print("=" * 50)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if model_name in ['random_forest', 'xgboost']:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'AUC Score': results['auc_score'],
                    'CV Mean': results['cv_mean'],
                    'CV Std': results['cv_std']
                })
            elif model_name == 'lstm':
                comparison_data.append({
                    'Model': 'LSTM',
                    'AUC Score': results['auc_score'],
                    'CV Mean': 'N/A',
                    'CV Std': 'N/A'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['AUC Score'].idxmax(), 'Model']
        best_auc = comparison_df['AUC Score'].max()
        
        print(f"\nBEST MODEL: {best_model} (AUC: {best_auc:.4f})")
        
        return comparison_df, best_model
    
    def create_model_evaluation_plots(self):
        """Create comprehensive evaluation plots"""
        print("\nCreating model evaluation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Machine Learning Model Evaluation Dashboard', fontsize=16)
        
        colors = ['blue', 'orange', 'green']
        model_names = ['Random Forest', 'XGBoost', 'LSTM']
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for i, (model_key, results) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            auc = results['auc_score']
            ax1.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{model_names[i]} (AUC = {auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Importance (Random Forest)
        ax2 = axes[0, 1]
        if 'random_forest' in self.results:
            rf_importance = self.results['random_forest']['feature_importance'].head(10)
            ax2.barh(rf_importance['feature'], rf_importance['importance'])
            ax2.set_title('Random Forest Feature Importance')
            ax2.set_xlabel('Importance')
        
        # 3. Feature Importance (XGBoost)
        ax3 = axes[0, 2]
        if 'xgboost' in self.results:
            xgb_importance = self.results['xgboost']['feature_importance'].head(10)
            ax3.barh(xgb_importance['feature'], xgb_importance['importance'])
            ax3.set_title('XGBoost Feature Importance')
            ax3.set_xlabel('Importance')
        
        # 4. Confusion Matrices
        for i, (model_key, results) in enumerate(self.results.items()):
            ax = axes[1, i]
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_names[i]} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # LSTM Training History (if available)
        if 'lstm' in self.results and 'history' in self.results['lstm']:
            self.plot_lstm_history()
    
    def plot_lstm_history(self):
        """Plot LSTM training history"""
        history = self.results['lstm']['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('LSTM Training History', fontsize=14)
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/lstm_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # This will be called from main analysis script
    pass