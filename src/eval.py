import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_and_evaluate_models

class SystemEvaluator:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def run_eval(self):
        results, data = train_and_evaluate_models(self.data_path)
        
        summary_rows = []
        for model_name, metrics in results.items():
            summary_rows.append([
                model_name,
                f"{metrics['Accuracy']*100:.2f}%",
                f"{metrics['Precision']*100:.2f}%",
                f"{metrics['Recall']*100:.2f}%",
                f"{metrics['F1-Score']*100:.2f}%"
            ])
            
        table_str = tabulate(
            summary_rows,
            headers=['Model Architecture', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            tablefmt='grid'
        )
        return table_str, results

if __name__ == '__main__':
    evaluator = SystemEvaluator('/workspace/scratch/vehicle_predictive_maintenance/data/vehicle_sensor_data.csv')
    table, _ = evaluator.run_eval()
    print("\n" + "="*70)
    print("VEHICLE PREDICTIVE MAINTENANCE & FAULT DETECTION EVALUATION")
    print("="*70)
    print(table)
