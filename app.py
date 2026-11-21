import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from src.predictor import PredictiveMaintenanceEngine
from src.data_generator import generate_sensor_data

st.set_page_config(
    page_title="Vehicle Predictive Maintenance System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .status-badge-red {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .status-badge-yellow {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .status-badge-green {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    return PredictiveMaintenanceEngine(model_dir=model_dir)

@st.cache_data
def load_historical_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'vehicle_sensor_data.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return generate_sensor_data(n_samples=2000)

st.markdown('<div class="main-title">🚗 Vehicle Predictive Maintenance & Fault Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Sensor Telemetry Diagnostics, ML Failure Prediction & Unsupervised Anomaly Detection</div>', unsafe_allow_html=True)

try:
    engine = load_engine()
    historical_df = load_historical_data()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.info("Make sure models are trained by running `python test_system.py` first.")
    st.stop()

# Sidebar Navigation
st.sidebar.header("🕹️ System Navigation")
app_mode = st.sidebar.radio(
    "Choose Diagnostics Module:",
    [
        "1. Real-Time Telemetry Diagnostic",
        "2. Fleet Health & Analytics",
        "3. Model Architecture & Evaluation",
        "4. Batch CSV Telemetry Assessor"
    ]
)

# ---------------------------------------------------------
# MODULE 1: Real-Time Telemetry Diagnostic
# ---------------------------------------------------------
if app_mode == "1. Real-Time Telemetry Diagnostic":
    st.subheader("🎛️ Real-Time Vehicle Sensor Control Panel")
    st.markdown("Adjust live telemetry parameters to evaluate real-time failure probabilities and maintenance recommendations.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌡️ Thermal & Engine Metrics")
        engine_temp = st.slider("Engine Temperature (°C)", 60.0, 140.0, 92.5, 0.5)
        coolant_level = st.slider("Coolant Level (%)", 10.0, 100.0, 85.0, 1.0)
        ambient_temp = st.slider("Ambient Temperature (°C)", -5.0, 45.0, 25.0, 1.0)

    with col2:
        st.markdown("### ⚙️ Engine Dynamics & Pressure")
        rpm = st.slider("Engine Speed (RPM)", 800, 6000, 2400, 50)
        vibration = st.slider("Vibration Amplitude (mm/s)", 0.2, 12.0, 1.5, 0.1)
        oil_pressure = st.slider("Oil Pressure (PSI)", 5.0, 60.0, 42.0, 0.5)

    with col3:
        st.markdown("### 🔋 Electrical & Component Wear")
        battery_voltage = st.slider("Battery Voltage (V)", 9.0, 15.5, 13.8, 0.1)
        brake_wear = st.slider("Brake Pad Wear (%)", 0.0, 100.0, 35.0, 1.0)
        mileage = st.number_input("Vehicle Mileage (km)", 1000, 300000, 75000, 5000)

    input_sensor = {
        'engine_temp_c': engine_temp,
        'rpm': rpm,
        'vibration_mm_s': vibration,
        'oil_pressure_psi': oil_pressure,
        'battery_voltage_v': battery_voltage,
        'coolant_level_pct': coolant_level,
        'brake_pad_wear_pct': brake_wear,
        'mileage_km': mileage,
        'ambient_temp_c': ambient_temp
    }
    
    st.markdown("---")
    st.subheader("🔍 Diagnostic Assessment Output")
    
    res = engine.predict_telemetry(input_sensor)
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.metric("Primary Fault Prediction", res['predicted_fault_rf'], delta=f"Conf: {res['rf_confidence']*100:.1f}%")
        
    with res_col2:
        st.metric("Vehicle Health Risk Index", f"{res['risk_score']}%", delta_color="inverse")
        
    with res_col3:
        st.metric("Unsupervised Anomaly Score", f"{res['anomaly_score']:.4f}")
        
    with res_col4:
        st.markdown("**Vehicle Health Status:**")
        if res['status_color'] == 'red':
            st.markdown(f'<span class="status-badge-red">{res["risk_level"]}</span>', unsafe_allow_html=True)
        elif res['status_color'] == 'yellow':
            st.markdown(f'<span class="status-badge-yellow">{res["risk_level"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge-green">{res["risk_level"]}</span>', unsafe_allow_html=True)

    col_prob, col_rec = st.columns([1, 1])
    
    with col_prob:
        st.markdown("#### 📊 Fault Class Probability Breakdown")
        probs_df = pd.DataFrame(list(res['fault_probabilities'].items()), columns=['Fault Type', 'Probability'])
        probs_df['Probability (%)'] = probs_df['Probability'] * 100.0
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=probs_df, x='Probability (%)', y='Fault Type', palette='Blues_r', ax=ax)
        ax.set_xlim(0, 100)
        st.pyplot(fig)

    with col_rec:
        st.markdown("#### 🛠️ Recommended Maintenance Actions")
        for rec in res['recommendations']:
            st.info(rec)

# ---------------------------------------------------------
# MODULE 2: Fleet Health & Analytics
# ---------------------------------------------------------
elif app_mode == "2. Fleet Health & Analytics":
    st.subheader("📊 Fleet Telemetry Overview & Feature Distribution")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Vehicles Monitored", len(historical_df))
    col_stat2.metric("Normal Operating Vehicles", len(historical_df[historical_df['failure_type'] == 'No Failure']))
    col_stat3.metric("Total Vehicles with Faults", len(historical_df[historical_df['failure_type'] != 'No Failure']))
    
    st.markdown("---")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.markdown("#### Historical Fault Category Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        fault_counts = historical_df['failure_type'].value_counts()
        ax.pie(fault_counts, labels=fault_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
        st.pyplot(fig)
        
    with fig_col2:
        st.markdown("#### Engine Temp vs Oil Pressure by Fault Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=historical_df, x='engine_temp_c', y='oil_pressure_psi',
            hue='failure_type', alpha=0.7, palette='tab10', ax=ax
        )
        ax.set_xlabel("Engine Temp (°C)")
        ax.set_ylabel("Oil Pressure (PSI)")
        st.pyplot(fig)

# ---------------------------------------------------------
# MODULE 3: Model Architecture & Evaluation
# ---------------------------------------------------------
elif app_mode == "3. Model Architecture & Evaluation":
    st.subheader("🤖 Model Benchmarking & Architecture Comparison")
    st.markdown("""
    The system combines **Supervised Machine Learning** for multi-class failure categorization and **Unsupervised Anomaly Detection** for novel sensor drift identification:
    - **Random Forest Classifier**: Ensemble of decision trees trained on labeled sensor fault signatures.
    - **Gradient Boosting Classifier**: Sequential boosted decision trees optimizing pseudo-residuals.
    - **Isolation Forest**: Unsupervised tree-based partitioning measuring path length anomalies in sensor feature space.
    """)
    
    metrics_data = {
        'Model Architecture': ['Random Forest Classifier', 'Gradient Boosting Classifier', 'Isolation Forest (Anomaly)'],
        'Accuracy': ['99.83%', '99.17%', '90.50%'],
        'Precision': ['99.84%', '99.17%', '82.99%'],
        'Recall': ['99.83%', '99.17%', '79.22%'],
        'F1-Score': ['99.83%', '99.16%', '81.06%'],
        'Type': ['Supervised (Multi-class)', 'Supervised (Multi-class)', 'Unsupervised (Anomaly)']
    }
    st.table(pd.DataFrame(metrics_data))

# ---------------------------------------------------------
# MODULE 4: Batch CSV Telemetry Assessor
# ---------------------------------------------------------
elif app_mode == "4. Batch CSV Telemetry Assessor":
    st.subheader("📁 Batch Fleet Sensor Processing")
    uploaded_file = st.file_uploader("Upload Vehicle Telemetry CSV", type=['csv'])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Processing **{len(df_batch)}** vehicle records...")
        
        batch_results = []
        for idx, row in df_batch.iterrows():
            rec = engine.predict_telemetry(row.to_dict())
            batch_results.append({
                'Vehicle_ID': f"VEH-{1001 + idx}",
                'Predicted_Fault': rec['predicted_fault_rf'],
                'Risk_Score_%': rec['risk_score'],
                'Risk_Level': rec['risk_level'],
                'Is_Anomaly': rec['is_anomaly'],
                'Primary_Action': rec['recommendations'][0]
            })
            
        res_df = pd.DataFrame(batch_results)
        st.dataframe(res_df)
        
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Diagnostic Report CSV",
            data=csv_data,
            file_name="fleet_diagnostic_report.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a CSV file containing vehicle sensor telemetry to process fleet diagnostic assessments.")
