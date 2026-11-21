import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import engineer_features, FEATURE_COLS

class PredictiveMaintenanceEngine:
    def __init__(self, model_dir: str = '/workspace/scratch/vehicle_predictive_maintenance/models'):
        self.model_dir = model_dir
        self.rf_model = joblib.load(os.path.join(model_dir, 'rf_classifier.joblib'))
        self.gb_model = joblib.load(os.path.join(model_dir, 'gb_classifier.joblib'))
        self.iso_forest = joblib.load(os.path.join(model_dir, 'isolation_forest.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

    def predict_telemetry(self, sensor_dict: dict) -> dict:
        """
        Input dict with keys:
        - engine_temp_c
        - rpm
        - vibration_mm_s
        - oil_pressure_psi
        - battery_voltage_v
        - coolant_level_pct
        - brake_pad_wear_pct
        - mileage_km
        - ambient_temp_c
        """
        df_input = pd.DataFrame([sensor_dict])
        df_engineered = engineer_features(df_input)
        X = df_engineered[FEATURE_COLS]
        
        # 1. Fault Classification Prediction
        rf_pred = self.rf_model.predict(X)[0]
        rf_probs = dict(zip(self.rf_model.classes_, self.rf_model.predict_proba(X)[0]))
        
        gb_pred = self.gb_model.predict(X)[0]
        gb_probs = dict(zip(self.gb_model.classes_, self.gb_model.predict_proba(X)[0]))
        
        # 2. Anomaly Detection via Isolation Forest
        iso_pred_raw = self.iso_forest.predict(X)[0] # -1: anomaly, 1: normal
        iso_score = float(-self.iso_forest.score_samples(X)[0]) # Higher score = more anomalous
        is_anomaly = bool(iso_pred_raw == -1)
        
        # 3. Calculate Overall Vehicle Health Risk Index (0 - 100%)
        # Base risk from non-normal probabilities + anomaly score
        no_fail_prob = rf_probs.get('No Failure', 0.8)
        base_risk = (1.0 - no_fail_prob) * 100.0
        anomaly_penalty = max(0.0, (iso_score - 0.4) * 100.0) if is_anomaly else 0.0
        
        risk_score = min(100.0, max(0.0, base_risk * 0.7 + anomaly_penalty * 0.3))
        
        # Risk level categorization
        if risk_score < 25.0:
            risk_level = "LOW (Healthy)"
            status_color = "green"
        elif risk_score < 60.0:
            risk_level = "MODERATE (Attention Needed)"
            status_color = "yellow"
        else:
            risk_level = "HIGH (Critical Fault)"
            status_color = "red"
            
        # 4. Generate Maintenance Recommendations
        recommendations = []
        primary_fault = rf_pred
        
        if primary_fault == 'Overheating':
            recommendations.append("🚨 CRITICAL: Engine temperature elevated beyond normal thresholds. Inspect coolant levels, radiator fan, and thermostat immediately.")
        elif primary_fault == 'Oil Leak':
            recommendations.append("⚠️ WARNING: Abnormally low oil pressure detected. Check engine oil levels and inspect gaskets/oil pan for leakage.")
        elif primary_fault == 'Bearing Fault':
            recommendations.append("🚨 CRITICAL: High mechanical vibration detected. Inspect engine bearings, drive shaft, and motor mounts for wear.")
        elif primary_fault == 'Battery Degradation':
            recommendations.append("⚠️ WARNING: Battery voltage dropping below operational standard (13.5V). Test alternator and battery health.")
        else:
            recommendations.append("✅ SYSTEM NORMAL: No immediate critical fault detected. Continue routine scheduled maintenance.")
            
        if sensor_dict.get('brake_pad_wear_pct', 0) > 80.0:
            recommendations.append("🔧 ROUTINE SERVICE: Brake pad wear exceeds 80%. Schedule replacement within 500 km.")
            
        if is_anomaly and primary_fault == 'No Failure':
            recommendations.append("🔍 ANOMALY DETECTED: Unsupervised Isolation Forest detected irregular sensor patterns not matching standard baseline profiles.")

        return {
            'predicted_fault_rf': rf_pred,
            'predicted_fault_gb': gb_pred,
            'rf_confidence': float(rf_probs[rf_pred]),
            'gb_confidence': float(gb_probs[gb_pred]),
            'is_anomaly': is_anomaly,
            'anomaly_score': round(iso_score, 4),
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'status_color': status_color,
            'fault_probabilities': {k: round(float(v), 4) for k, v in rf_probs.items()},
            'recommendations': recommendations
        }

if __name__ == '__main__':
    engine = PredictiveMaintenanceEngine()
    test_sample = {
        'engine_temp_c': 118.5,
        'rpm': 3200,
        'vibration_mm_s': 1.8,
        'oil_pressure_psi': 42.0,
        'battery_voltage_v': 13.8,
        'coolant_level_pct': 32.0,
        'brake_pad_wear_pct': 45.0,
        'mileage_km': 85000,
        'ambient_temp_c': 28.0
    }
    res = engine.predict_telemetry(test_sample)
    print("Test Sample Prediction Output:")
    import json
    print(json.dumps(res, indent=2))
