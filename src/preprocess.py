import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    'engine_temp_c',
    'rpm',
    'vibration_mm_s',
    'oil_pressure_psi',
    'battery_voltage_v',
    'coolant_level_pct',
    'brake_pad_wear_pct',
    'mileage_km',
    'ambient_temp_c',
    'temp_diff_ambient',
    'rpm_vibration_ratio',
    'wear_rate'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain-specific feature engineering ratios."""
    df_feat = df.copy()
    
    # Feature engineering: thermal stress relative to ambient
    df_feat['temp_diff_ambient'] = df_feat['engine_temp_c'] - df_feat['ambient_temp_c']
    
    # RPM to Vibration stress index
    df_feat['rpm_vibration_ratio'] = df_feat['rpm'] * df_feat['vibration_mm_s'] / 1000.0
    
    # Brake wear rate per 1000 km
    df_feat['wear_rate'] = df_feat['brake_pad_wear_pct'] / (df_feat['mileage_km'] / 1000.0 + 1.0)
    
    return df_feat

def prepare_datasets(data_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_path)
    df_feat = engineer_features(df)
    
    X = df_feat[FEATURE_COLS]
    y_binary = df_feat['failure_occurred']
    y_multi = df_feat['failure_type']
    
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi, test_size=test_size, random_state=random_state, stratify=y_multi
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'y_train_bin': y_train_bin,
        'y_test_bin': y_test_bin,
        'y_train_multi': y_train_multi,
        'y_test_multi': y_test_multi,
        'scaler': scaler,
        'feature_names': FEATURE_COLS
    }
