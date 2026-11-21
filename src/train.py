import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocess import prepare_datasets

def train_and_evaluate_models(data_path: str, model_dir: str = '/workspace/scratch/vehicle_predictive_maintenance/models'):
    os.makedirs(model_dir, exist_ok=True)
    data = prepare_datasets(data_path)
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train_bin, y_test_bin = data['y_train_bin'], data['y_test_bin']
    y_train_multi, y_test_multi = data['y_train_multi'], data['y_test_multi']
    scaler = data['scaler']
    
    # ---------------------------------------------------------
    # 1. Supervised Models: Multi-class Fault Type Classifiers
    # ---------------------------------------------------------
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train_multi)
    
    gb_model = GradientBoostingClassifier(n_estimators=120, max_depth=5, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train_multi)
    
    # Predict multi-class
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test_multi, rf_pred)
    gb_acc = accuracy_score(y_test_multi, gb_pred)
    
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test_multi, rf_pred, average='weighted')
    gb_p, gb_r, gb_f1, _ = precision_recall_fscore_support(y_test_multi, gb_pred, average='weighted')
    
    # ---------------------------------------------------------
    # 2. Unsupervised Model: Isolation Forest for Anomaly Detection
    # ---------------------------------------------------------
    iso_forest = IsolationForest(contamination=0.25, random_state=42)
    iso_forest.fit(X_train)
    
    iso_pred_raw = iso_forest.predict(X_test)
    iso_anomalies = np.where(iso_pred_raw == -1, 1, 0)
    
    iso_acc = accuracy_score(y_test_bin, iso_anomalies)
    iso_p, iso_r, iso_f1, _ = precision_recall_fscore_support(y_test_bin, iso_anomalies, average='binary')
    
    # Save artifacts
    joblib.dump(rf_model, os.path.join(model_dir, 'rf_classifier.joblib'))
    joblib.dump(gb_model, os.path.join(model_dir, 'gb_classifier.joblib'))
    joblib.dump(iso_forest, os.path.join(model_dir, 'isolation_forest.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    results = {
        'RandomForest': {
            'Accuracy': rf_acc,
            'Precision': rf_p,
            'Recall': rf_r,
            'F1-Score': rf_f1,
            'ModelObj': rf_model
        },
        'GradientBoosting': {
            'Accuracy': gb_acc,
            'Precision': gb_p,
            'Recall': gb_r,
            'F1-Score': gb_f1,
            'ModelObj': gb_model
        },
        'IsolationForest': {
            'Accuracy': iso_acc,
            'Precision': iso_p,
            'Recall': iso_r,
            'F1-Score': iso_f1,
            'ModelObj': iso_forest
        }
    }
    
    return results, data

if __name__ == '__main__':
    data_file = '/workspace/scratch/vehicle_predictive_maintenance/data/vehicle_sensor_data.csv'
    res, _ = train_and_evaluate_models(data_file)
    print("=== MODEL EVALUATION SUMMARY ===")
    for model_name, metrics in res.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if k != 'ModelObj':
                print(f"  {k}: {v:.4f}")
