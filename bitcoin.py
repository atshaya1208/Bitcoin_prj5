import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page layout
st.set_page_config(layout="wide", page_title="Forecasting Model Evaluation")

# Keep your model architecture intact in case you want to use it later
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=3):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- FIXED DATA LOADER ---
@st.cache_data
def load_saved_data():
    # Only load what your app actually uses for the charts
    y_test_orig = np.load(r'C:\Users\mbhar\OneDrive\Desktop\project5\env\Scripts\y_true_orig.npy')
    y_pred_orig = np.load(r'C:\Users\mbhar\OneDrive\Desktop\project5\env\Scripts\y_pred_orig.npy')
    return y_test_orig, y_pred_orig

y_true_orig, y_pred_orig = load_saved_data()
horizons = ['1-Day', '3-Day', '7-Day']

st.sidebar.header("Model Controls")
selected_model = st.sidebar.selectbox("Select Model Architecture", ["LSTM", "1D-CNN", "Transformer"])

st.title(" Model Evaluation & Visualization Dashboard")
st.markdown("Inspect performance metrics, error distributions, and future horizon comparison curves.")

st.header(" Test Set Metrics Evaluation")

metrics_summary = {}
for i, horizon in enumerate(horizons):
    mae = mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i])
    rmse = np.sqrt(mean_squared_error(y_true_orig[:, i], y_pred_orig[:, i]))
    mape = np.mean(np.abs((y_true_orig[:, i] - y_pred_orig[:, i]) / (y_true_orig[:, i] + 1e-5))) * 100
    metrics_summary[horizon] = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}

metrics_df = pd.DataFrame(metrics_summary).T
st.table(metrics_df)

st.header(" Model Visualizations")
tabs = st.tabs(["Actual vs Predicted", "Error Distributions", "Horizon Comparison"])

with tabs[0]:
    st.subheader("Actual vs Predicted Price Curves")
    horizon_idx = st.selectbox("Select Horizon for Price Curve:", range(len(horizons)), format_func=lambda x: horizons[x])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true_orig[:, horizon_idx], label="Actual Price", color="blue", alpha=0.7)
    ax.plot(y_pred_orig[:, horizon_idx], label="Predicted Price", color="orange", linestyle="--", alpha=0.9)
    ax.set_title(f"Price Curve Sequence Comparison — {horizons[horizon_idx]} Horizon")
    ax.set_xlabel("Test Samples")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Error Distribution Plots (Residuals)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for i, horizon in enumerate(horizons):
        residuals = y_true_orig[:, i] - y_pred_orig[:, i]
        sns.histplot(residuals, kde=True, ax=axes[i], color="purple", bins=20)
        axes[i].axvline(x=0, color='red', linestyle=':')
        axes[i].set_title(f"Errors for {horizon}")
        axes[i].set_xlabel("Residual Error")
        
    plt.tight_layout()
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Forecast Horizon Tracking Performance")
    sample_idx = st.slider("Select an Individual Test Sample to view forecasted trajectory:", 0, len(y_true_orig)-1, 0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(horizons, y_true_orig[sample_idx, :], marker='o', label="Actual Path", color='black', linewidth=2)
    ax.plot(horizons, y_pred_orig[sample_idx, :], marker='x', label="Predicted Path", color='crimson', linestyle='--')
    ax.set_title(f"Trajectory Track for Sample #{sample_idx}")
    ax.set_ylabel("Price Valuation")
    ax.legend()
    st.pyplot(fig)