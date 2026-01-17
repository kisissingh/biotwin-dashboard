import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="BioTwin: Golden Batch System", layout="wide")

st.title("🧬 BioTwin: Real-Time Release (RTR) Dashboard")
st.markdown("**Status:** FDA 21 CFR Part 11 Compliant | **System:** Bioreactor B-204")

# --- 1. THE "FAKE FACTORY" (Data Gen) ---
@st.cache_data # Caches the data so it doesn't reload every time
def generate_data():
    time_steps = np.linspace(0, 100, 100)
    
    # Generate 50 Good Batches (Training Data)
    good_batches = []
    for i in range(50):
        biomass = 10.0 / (1 + np.exp(-0.1 * (time_steps - 50))) + np.random.normal(0, 0.1, 100)
        do_level = 100 - (biomass * 8) + np.random.normal(0, 0.5, 100)
        good_batches.append(np.column_stack((biomass, do_level)))
    
    # Generate 1 Bad Batch (Sensor Failure at 60h)
    bad_biomass = 10.0 / (1 + np.exp(-0.1 * (time_steps - 50))) + np.random.normal(0, 0.1, 100)
    bad_do = 100 - (bad_biomass * 8) + np.random.normal(0, 0.5, 100)
    bad_do[60:] = 120 # MASSIVE SENSOR SPIKE
    bad_batch = pd.DataFrame({'Time': time_steps, 'Biomass': bad_biomass, 'DO': bad_do})
    
    return np.vstack(good_batches), bad_batch

# Load Data
X_train, bad_batch = generate_data()

# --- 2. THE "BRAIN" (Model Training) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=1)
pca.fit(X_train_scaled)

# Calculate Anomaly Scores for the Bad Batch
X_bad_scaled = scaler.transform(bad_batch[['Biomass', 'DO']])
X_bad_pca = pca.transform(X_bad_scaled)
X_bad_recon = pca.inverse_transform(X_bad_pca)
recon_error = np.sum(np.square(X_bad_scaled - X_bad_recon), axis=1)

# --- 3. THE "CONTROL TOWER" (Dashboard Interface) ---

# Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")
current_time = st.sidebar.slider("Batch Time (Hours)", 0, 99, 0)
show_golden = st.sidebar.checkbox("Show Golden Tunnel", value=True)

# Get Data for Current Time
current_row = bad_batch.iloc[current_time]
current_score = recon_error[current_time]

# Top Metrics Row
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Dissolved Oxygen", f"{current_row['DO']:.2f}%", delta="-0.5%" if current_time < 60 else "-100%")
kpi2.metric("Biomass Density", f"{current_row['Biomass']:.2f} g/L")
kpi3.metric("Anomaly Score", f"{current_score:.2f}", delta_color="inverse")

# --- SMART LOGIC: FIND FIRST FAILURE TIME ---
failure_indices = np.where(recon_error > 3.0)[0]
first_failure_time = failure_indices[0] if len(failure_indices) > 0 else 999

# --- MAIN INTERFACE LOGIC ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Live Sensor Data (Dissolved Oxygen)")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 1. THE GHOST LINE (The Future Prediction)
    # We plot the WHOLE batch faintly, so you can see where the spike was going.
    ax.plot(bad_batch['Time'], bad_batch['DO'], color='red', linestyle=':', alpha=0.3, label='Predicted Trajectory')
    
    # 2. THE REALITY (The Stopped Process)
    # We only plot the solid line up to the stop point.
    display_limit = min(current_time, first_failure_time) if current_time > first_failure_time else current_time
    ax.plot(bad_batch['Time'][:display_limit+1], bad_batch['DO'][:display_limit+1], color='red', linewidth=2, label='Actual Process')
    
    # The Golden Tunnel
    if show_golden:
        ax.fill_between(bad_batch['Time'], 80, 110, color='gray', alpha=0.2, label='Golden Tunnel')
    
    # The "Kill" Marker
    if current_time >= first_failure_time:
        ax.axvline(first_failure_time, color='black', linestyle='--', label='Auto-Shutoff')
        ax.scatter(bad_batch['Time'][first_failure_time], bad_batch['DO'][first_failure_time], color='black', s=100, zorder=5, marker='X')
    
    ax.set_ylim(-10, 140) # Increased limit to see the spike
    ax.legend(loc='lower left')
    st.pyplot(fig)

with col2:
    st.subheader("🧠 Digital Twin Diagnostics")
    
    # Status Logic (FIXED THE BUG HERE)
    if current_time < first_failure_time:
        st.success("✅ Process Within Control Limits")
        st.metric("System Status", "RUNNING", delta="Optimal")
    else:
        st.error(f"🚨 CRITICAL FAILURE DETECTED at Hour {first_failure_time}")
        st.warning("⚠️ AUTOMATED ACTION: Feed Pump Stopped.")
        # Fixed line: Moved "HALTED" to the 'delta' argument
        st.metric("System Status", "TERMINATED", delta="HALTED", delta_color="inverse")

    # The Anomaly Chart (With Ghost Line)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    # Ghost line for anomaly score
    ax2.plot(bad_batch['Time'], recon_error, color='purple', linestyle=':', alpha=0.3)
    # Solid line for actual score
    ax2.plot(bad_batch['Time'][:display_limit+1], recon_error[:display_limit+1], color='purple', linewidth=2, label='Deviation Score')
    ax2.axhline(3.0, color='red', linestyle='--', label='Threshold')
    st.pyplot(fig2)

# Footer
st.caption("BioTwin v2.1 | Automated Process Control (APC) Enabled")
