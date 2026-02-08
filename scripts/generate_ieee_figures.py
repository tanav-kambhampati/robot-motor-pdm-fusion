"""
IEEE-Compatible Figure Generator for Motor Predictive Maintenance Paper
Generates individual figures with white backgrounds and large fonts for print publication
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import xgboost as xgb
from data_loader import MotorDataLoader
import warnings
warnings.filterwarnings('ignore')

# IEEE-compatible styling - large fonts for two-column format
plt.style.use('default')
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 17,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Professional color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'success': '#28A745',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
}

def create_ieee_feature_importance(feature_importance, save_path='plots/ieee_feature_importance.png'):
    """Create feature importance bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    top_features = feature_importance.head(10)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)
    
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=16, fontweight='bold')
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=18)
    ax.set_xlabel('Importance Score', fontsize=20)
    ax.set_title('Feature Importance Analysis', fontsize=22, fontweight='bold', pad=15)
    ax.set_xlim(0, max(top_features['importance']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")


def create_ieee_correlation_heatmap(X, feature_names, save_path='plots/ieee_correlation_heatmap.png'):
    """Create correlation heatmap"""
    correlation_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                cmap=cmap, center=0, square=True,
                xticklabels=feature_names, yticklabels=feature_names,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                annot_kws={'size': 16}, linewidths=0.5, linecolor='white',
                ax=ax)
    
    ax.set_title('Feature Correlation Analysis', fontsize=22, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")


def create_ieee_3d_pca(X_scaled, y, save_path='plots/ieee_3d_pca.png'):
    """Create 3D PCA visualization"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    normal_mask = y == 0
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], X_pca[normal_mask, 2],
              c=COLORS['primary'], s=15, alpha=0.5, label='Normal', marker='o')
    
    anomaly_mask = y == 1
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], X_pca[anomaly_mask, 2],
              c=COLORS['danger'], s=40, alpha=0.8, marker='X', label='Anomalies')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=18, labelpad=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=18, labelpad=10)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=18, labelpad=10)
    ax.set_title('3D Feature Space (PCA)', fontsize=22, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=18, frameon=True, facecolor='white', edgecolor='gray')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)
    
    # Add extra right margin for z-axis label
    plt.subplots_adjust(right=0.82)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")
    return pca


def create_ieee_learning_curves(X, y, save_path='plots/ieee_learning_curves.png'):
    """Create learning curves - single combined plot"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    colors = [COLORS['success'], COLORS['danger']]
    
    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        ax.set_facecolor('white')
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color=colors[i])
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color=colors[i])
        
        ax.plot(train_sizes, train_mean, 'o-', color=colors[i], 
               label='Training', linewidth=2.5, markersize=8)
        ax.plot(train_sizes, val_mean, 's--', color=colors[i], 
               label='Validation', linewidth=2.5, markersize=8, alpha=0.8)
        
        ax.set_title(f'{name}', fontsize=20, fontweight='bold')
        ax.set_xlabel('Training Examples', fontsize=18)
        ax.set_ylabel('AUC Score', fontsize=18)
        ax.legend(loc='lower right', fontsize=16, frameon=True, facecolor='white')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.05)
    
    plt.suptitle('Model Learning Curves', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")


# === SPLIT INDIVIDUAL FIGURES ===

def create_ieee_roc_curves(y_test, rf_pred_proba, xgb_pred_proba, save_path='plots/ieee_roc_curves.png'):
    """Create ROC curves as standalone figure"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)
    
    ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC={rf_auc:.3f})', 
             color=COLORS['success'], linewidth=3)
    ax.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC={xgb_auc:.3f})', 
             color=COLORS['warning'], linewidth=3)
    ax.plot([0, 1], [0, 1], '--', color=COLORS['neutral'], alpha=0.8, linewidth=2)
    
    ax.fill_between(rf_fpr, rf_tpr, alpha=0.15, color=COLORS['success'])
    ax.fill_between(xgb_fpr, xgb_tpr, alpha=0.15, color=COLORS['warning'])
    
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title('ROC Curves Comparison', fontsize=22, fontweight='bold')
    ax.legend(loc='lower right', fontsize=18, frameon=True, facecolor='white')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")


def create_ieee_confusion_matrix(y_test, rf_pred, save_path='plots/ieee_confusion_matrix.png'):
    """Create confusion matrix as standalone figure"""
    fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
    ax.set_facecolor('white')
    
    cm = confusion_matrix(y_test, rf_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=22, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                    ha="center", va="center", color=text_color, 
                    fontweight='bold', fontsize=20)
    
    ax.set_ylabel('True Label', fontsize=20)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=18)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=18)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path}")


def main():
    print("=" * 60)
    print("IEEE-COMPATIBLE FIGURE GENERATOR")
    print("Generating individual figures with large fonts")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading motor sensor data...")
    loader = MotorDataLoader()
    combined_data = loader.combine_all_data()
    anomaly_data = loader.detect_anomalies(method='iqr')
    
    print(f"   Loaded {len(anomaly_data)} samples")
    print(f"   Anomaly rate: {(anomaly_data['is_anomaly'].sum() / len(anomaly_data) * 100):.2f}%")
    
    # Prepare features
    print("\n[2/6] Preparing features...")
    feature_cols = ['temperature', 'voltage', 'position', 'relative_time']
    
    anomaly_data['temp_rolling_mean'] = anomaly_data.groupby(['session', 'motor_id'])['temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    anomaly_data['voltage_rolling_std'] = anomaly_data.groupby(['session', 'motor_id'])['voltage'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    
    le_session = LabelEncoder()
    le_motor = LabelEncoder()
    
    anomaly_data['session_encoded'] = le_session.fit_transform(anomaly_data['session'])
    anomaly_data['motor_encoded'] = le_motor.fit_transform(anomaly_data['motor_id'])
    
    feature_cols.extend(['temp_rolling_mean', 'voltage_rolling_std', 'session_encoded', 'motor_encoded'])
    
    X = anomaly_data[feature_cols].fillna(0)
    y = anomaly_data['is_anomaly'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Created {len(feature_cols)} features")
    print(f"   Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train models
    print("\n[3/6] Training models...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                   random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='auc')
    xgb_model.fit(X_train, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    print(f"   Random Forest AUC: {rf_auc:.4f}")
    print(f"   XGBoost AUC: {xgb_auc:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Generate figures
    print("\n[4/6] Generating IEEE-compatible figures...")
    
    print("\n   Figure 1: Feature Importance")
    create_ieee_feature_importance(feature_importance)
    
    print("   Figure 2: Correlation Heatmap")
    create_ieee_correlation_heatmap(X_train_scaled, feature_cols)
    
    print("   Figure 3: 3D PCA Visualization")
    create_ieee_3d_pca(X_train_scaled[:5000], y_train.iloc[:5000].values)
    
    print("   Figure 4: Learning Curves")
    create_ieee_learning_curves(X_train_scaled, y_train)
    
    print("   Figure 5: ROC Curves (standalone)")
    create_ieee_roc_curves(y_test, rf_pred_proba, xgb_pred_proba)
    
    print("   Figure 6: Confusion Matrix (standalone)")
    create_ieee_confusion_matrix(y_test, rf_pred)
    
    print("\n" + "=" * 60)
    print("SUCCESS! All IEEE-compatible figures generated.")
    print("=" * 60)
    print("\nGenerated files in plots/:")
    print("  - ieee_feature_importance.png")
    print("  - ieee_correlation_heatmap.png")
    print("  - ieee_3d_pca.png")
    print("  - ieee_learning_curves.png")
    print("  - ieee_roc_curves.png (NEW - standalone)")
    print("  - ieee_confusion_matrix.png (NEW - standalone)")
    print("\nAll figures have:")
    print("  - Large fonts (18-22pt)")
    print("  - White backgrounds")
    print("  - 300 DPI resolution")
    
    return True


if __name__ == "__main__":
    main()
