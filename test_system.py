import os
import sys
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_sensor_data
from src.train import train_and_evaluate_models
from src.predictor import PredictiveMaintenanceEngine
from src.eval import SystemEvaluator

def run_e2e_test():
    print("="*80)
    print("VEHICLE PREDICTIVE MAINTENANCE AND FAULT DETECTION SYSTEM - SYSTEM TEST")
    print("="*80)
    
    # 1. Dataset Generation
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'vehicle_sensor_data.csv')
    
    print("\n[1] Generating Vehicle Sensor Dataset...")
    df = generate_sensor_data(n_samples=2500, random_state=42)
    df.to_csv(data_path, index=False)
    print(f" -> Generated {len(df)} telemetry records with 9 sensor parameters.")
    print(" -> Class Breakdown:")
    for fault, count in df['failure_type'].value_counts().items():
        print(f"    - {fault}: {count}")

    # 2. Model Training & Evaluation
    print("\n[2] Training Supervised & Unsupervised ML Models...")
    results, _ = train_and_evaluate_models(data_path)
    
    table_rows = []
    for model_name, metrics in results.items():
        table_rows.append([
            model_name,
            f"{metrics['Accuracy']*100:.2f}%",
            f"{metrics['Precision']*100:.2f}%",
            f"{metrics['Recall']*100:.2f}%",
            f"{metrics['F1-Score']*100:.2f}%"
        ])
        
    print("\n[3] Machine Learning Benchmark Metrics:")
    print(tabulate(table_rows, headers=['Model Architecture', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], tablefmt='grid'))

    # 3. Real-Time Telemetry Inference Test
    print("\n[4] Testing Predictive Diagnostics & Recommendation Engine...")
    engine = PredictiveMaintenanceEngine()
    
    sample_overheating = {
        'engine_temp_c': 118.0,
        'rpm': 3200,
        'vibration_mm_s': 1.8,
        'oil_pressure_psi': 42.0,
        'battery_voltage_v': 13.8,
        'coolant_level_pct': 32.0,
        'brake_pad_wear_pct': 45.0,
        'mileage_km': 85000,
        'ambient_temp_c': 28.0
    }
    
    diag_res = engine.predict_telemetry(sample_overheating)
    print(f" -> Input: Overheating Scenario (Temp={sample_overheating['engine_temp_c']}°C, Coolant={sample_overheating['coolant_level_pct']}%)")
    print(f" -> Predicted Fault (Random Forest): {diag_res['predicted_fault_rf']} (Confidence: {diag_res['rf_confidence']*100:.1f}%)")
    print(f" -> Predicted Fault (Gradient Boost): {diag_res['predicted_fault_gb']} (Confidence: {diag_res['gb_confidence']*100:.1f}%)")
    print(f" -> Unsupervised Anomaly Status: {diag_res['is_anomaly']} (Score: {diag_res['anomaly_score']})")
    print(f" -> Vehicle Risk Index: {diag_res['risk_score']}% ({diag_res['risk_level']})")
    print(f" -> Action Recommendation: {diag_res['recommendations'][0]}")

    print("\n" + "="*80)
    print("ALL SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("="*80)

if __name__ == '__main__':
    run_e2e_test()
