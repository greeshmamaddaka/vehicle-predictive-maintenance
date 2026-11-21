import os
import numpy as np
import pandas as pd

def generate_sensor_data(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # Generate baseline normal operational metrics
    mileage_km = np.random.uniform(5000, 200000, n_samples)
    ambient_temp = np.random.uniform(10, 40, n_samples)
    
    # Baseline normal telemetry
    engine_temp = np.random.normal(90, 4, n_samples) + (ambient_temp * 0.1)
    rpm = np.random.normal(2400, 400, n_samples)
    vibration = np.random.gamma(2.0, 0.7, n_samples) # mm/s
    oil_pressure = np.random.normal(40, 5, n_samples) # psi
    battery_voltage = np.random.normal(13.8, 0.3, n_samples) # Volts
    coolant_level = np.random.uniform(70, 100, n_samples) - (mileage_km / 10000) * 0.5
    coolant_level = np.clip(coolant_level, 20, 100)
    brake_wear = np.clip((mileage_km / 2000) + np.random.normal(0, 5, n_samples), 0, 100)
    
    failure_type = ['No Failure'] * n_samples
    failure_occurred = np.zeros(n_samples, dtype=int)
    
    # Inject failure patterns (approx 25% total failure rate)
    for i in range(n_samples):
        r = np.random.rand()
        if r < 0.08:
            # Overheating fault
            engine_temp[i] += np.random.uniform(20, 35)
            coolant_level[i] -= np.random.uniform(30, 50)
            coolant_level[i] = max(10, coolant_level[i])
            failure_type[i] = 'Overheating'
            failure_occurred[i] = 1
        elif r < 0.14:
            # Oil Leak / Low Pressure fault
            oil_pressure[i] -= np.random.uniform(18, 28)
            engine_temp[i] += np.random.uniform(8, 15)
            failure_type[i] = 'Oil Leak'
            failure_occurred[i] = 1
        elif r < 0.20:
            # Mechanical / Bearing Vibration fault
            vibration[i] += np.random.uniform(4.5, 9.5)
            rpm[i] += np.random.uniform(300, 800)
            failure_type[i] = 'Bearing Fault'
            failure_occurred[i] = 1
        elif r < 0.25:
            # Battery / Electrical Degradation
            battery_voltage[i] -= np.random.uniform(1.8, 3.2)
            failure_type[i] = 'Battery Degradation'
            failure_occurred[i] = 1

    df = pd.DataFrame({
        'engine_temp_c': np.round(engine_temp, 2),
        'rpm': np.round(rpm, 0),
        'vibration_mm_s': np.round(vibration, 2),
        'oil_pressure_psi': np.round(oil_pressure, 2),
        'battery_voltage_v': np.round(battery_voltage, 2),
        'coolant_level_pct': np.round(coolant_level, 1),
        'brake_pad_wear_pct': np.round(brake_wear, 1),
        'mileage_km': np.round(mileage_km, 0),
        'ambient_temp_c': np.round(ambient_temp, 1),
        'failure_type': failure_type,
        'failure_occurred': failure_occurred
    })
    
    return df

if __name__ == '__main__':
    os.makedirs('/workspace/scratch/vehicle_predictive_maintenance/data', exist_ok=True)
    df = generate_sensor_data(n_samples=3000)
    out_path = '/workspace/scratch/vehicle_predictive_maintenance/data/vehicle_sensor_data.csv'
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic sensor data: {df.shape[0]} rows, saved to {out_path}")
    print(df['failure_type'].value_counts())
