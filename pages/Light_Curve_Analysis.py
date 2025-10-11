import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
import base64
import plotly.express as px
import os
import joblib
from scipy.fft import fft
from astropy.io import fits
import time
import shutil
from styling import add_advanced_loading_animation, load_custom_styling_back


add_advanced_loading_animation()
load_custom_styling_back()
# --- Page Config & Styling ---
st.set_page_config(layout="wide")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base66.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_png_as_page_bg('background.jpg')
except Exception:
    pass

# --- PYTORCH MODEL DEFINITION ---
class ExoCNN(nn.Module):
    def __init__(self, input_size):
        super(ExoCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        dummy_input = torch.zeros(1, 1, input_size)
        x = self.pool(self.relu(self.bn2(self.conv2(self.pool(self.relu(self.bn1(self.conv1(dummy_input))))))))
        self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# --- Artifact Loading ---
N_POINTS_BASE = 1000
INPUT_SIZE = N_POINTS_BASE + (N_POINTS_BASE // 2)

@st.cache_resource
def load_artifacts():
    artifacts_path = "Light_Curve_Models_Artifacts"
    model_path = os.path.join(artifacts_path, "best_exoplanet_model_v4.pt")
    scaler_path = os.path.join(artifacts_path, "standard_scaler.pkl")

    try:
        model = ExoCNN(input_size=INPUT_SIZE)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: '{os.path.basename(model_path)}' not found in '{artifacts_path}' folder.")
        return None, None

    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: '{os.path.basename(scaler_path)}' not found in '{artifacts_path}' folder.")
        return None, None

    return model, scaler

model, scaler = load_artifacts()

# --- Preprocessing Function ---
def preprocess_fits_for_prediction(fits_file, n_points=1000):
    try:
        with fits.open(fits_file) as hdul:
            flux = None
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None and hasattr(hdu, "columns"):
                    colnames = [c.upper() for c in hdu.columns.names]
                    if "PDCSAP_FLUX" in colnames:
                        flux = hdu.data["PDCSAP_FLUX"]
                        break
            if flux is None:
                for hdu in hdul:
                    if hasattr(hdu, "data") and hdu.data is not None and hasattr(hdu, "columns"):
                        colnames = [c.upper() for c in hdu.columns.names]
                        if "SAP_FLUX" in colnames:
                            flux = hdu.data["SAP_FLUX"]
                            break

        if flux is None:
            st.error("Could not find PDCSAP_FLUX or SAP_FLUX in FITS file.")
            return None, None, None

        lc_raw = lk.LightCurve(flux=flux).remove_nans()
        flux = lc_raw.flux.value

        if len(flux) < 100:
            st.error("Not enough valid data points (< 100) in the FITS file.")
            return None, None, None

        # --- CORRECTED PREPROCESSING LOGIC ---
        # 1. Resample the raw flux
        idx = np.linspace(0, len(flux) - 1, n_points).astype(int)
        flux_resampled = flux[idx]

        # 2. Perform initial normalization (important for FFT consistency)
        flux_normalized = (flux_resampled - np.mean(flux_resampled)) / (np.std(flux_resampled) + 1e-6)

        # 3. Calculate FFT features from the normalized flux
        fft_features = np.abs(fft(flux_normalized))[:n_points // 2]
        
        # 4. Combine normalized flux and FFT features BEFORE scaling
        combined_features = np.concatenate([flux_normalized, fft_features])
        
        # 5. NOW, use the scaler on the combined 1500 features
        final_features_scaled = scaler.transform(combined_features.reshape(1, -1))[0]
        
        # 6. Convert to a PyTorch tensor
        final_tensor = torch.tensor(final_features_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # For plotting, we can use the initially normalized curve
        lc_flat = lk.LightCurve(flux=flux_normalized) 
        
        return lc_raw, lc_flat, final_tensor

    except Exception as e:
        st.error(f"Failed to process FITS file: {e}")
        return None, None, None

# --- Main Page UI ---
st.title("📈 Light Curve Analysis")
if model is None or scaler is None:
    st.stop()

page = st.radio(
    "Choose a view:",
    ["Analysis", "⚙️ Admin & Model Management"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# --- ANALYSIS VIEW ---
if page == "Analysis":
    st.write("Upload a FITS file to analyze it with the `ExoCNN` deep learning model.")
    uploaded_fits = st.file_uploader("Upload or Drag and Drop your FITS file here", type=["fits", "fit"])

    if uploaded_fits is not None:
        lc_raw, lc_flat, flux_tensor = preprocess_fits_for_prediction(uploaded_fits, n_points=N_POINTS_BASE)
        if flux_tensor is not None:
            with st.spinner("Running prediction..."):
                with torch.no_grad():
                    final_prob = model(flux_tensor).item()

            verdict = "Confirmed EXO-Planet" if final_prob >= 0.5 else "False Positive"

            st.header("Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                if verdict == "Confirmed EXO-Planet":
                    st.success(f"**Verdict:** {verdict}")
                else:
                    st.error(f"**Verdict:** {verdict}")
            with col2:
                confidence = final_prob if verdict == "Confirmed EXO-Planet" else 1 - final_prob
                st.metric("Model Confidence Score", f"{confidence:.2%}")

            st.progress(confidence)

            st.header("Visualizations")
            plot_tabs = st.tabs(["Raw Light Curve", "Processed Light Curve", "Periodogram", "Folded Light Curve"])
            with plot_tabs[0]:
                st.subheader("Raw Light Curve")
                with st.spinner("Generating raw light curve..."):
                    fig_raw = px.line(x=lc_raw.time.value, y=lc_raw.flux.value, title="Raw Light Curve from FITS File")
                    st.plotly_chart(fig_raw, use_container_width=True)

            with plot_tabs[1]:
                st.subheader("Processed (Scaled) Light Curve")
                with st.spinner("Generating processed light curve..."):
                    fig_processed = px.line(x=lc_flat.time.value, y=lc_flat.flux.value, title="Scaled Light Curve")
                    st.plotly_chart(fig_processed, use_container_width=True)

            with plot_tabs[2]:
                st.subheader("Periodogram (Signal Search)")
                with st.spinner("Generating periodogram..."):
                    try:
                        periodogram = lc_flat.to_periodogram(method='bls')
                        fig_periodogram = px.line(x=periodogram.period.value, y=periodogram.power.value, title="Box-Least-Squares (BLS) Periodogram")
                        st.plotly_chart(fig_periodogram, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate a periodogram: {e}")

            with plot_tabs[3]:
                st.subheader("Folded Light Curve (at highest power)")
                with st.spinner("Searching for periodic signals and folding light curve..."):
                    try:
                        periodogram = lc_flat.to_periodogram(method='bls')
                        best_period = periodogram.period_at_max_power
                        folded_lc = lc_flat.fold(period=best_period)
                        fig_folded = px.scatter(x=folded_lc.time.value, y=folded_lc.flux.value, title=f"Light Curve Folded at Best Period ({best_period.value:.4f} days)")
                        st.plotly_chart(fig_folded, use_container_width=True)
                        st.success(f"The strongest periodic signal was found at {best_period.value:.4f} days.")
                    except Exception as e:
                        st.warning(f"Could not generate a folded light curve. This usually means no strong periodic signal was found in the data.")

# --- ADMIN VIEW ---
elif page == "⚙️ Admin & Model Management":
    st.header("⚙️ Admin & Model Management")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.warning("This section is restricted. Please enter the password to unlock. The Password is : 2030")
        password = st.text_input("Admin Password", type="password", key="lc_admin_password")
        if st.button("Unlock"):
            if password == "2030":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.success("Admin Mode Unlocked.")

        with st.expander("⬆️ Upload New Light Curve Model"):
            st.write("Manually replace the current PyTorch model file (`best_exoplanet_model_V2.pt`).")
            artifacts_path = "Light_Curve_Models_Artifacts"
            new_model_file = st.file_uploader("Upload best_exoplanet_model.pt", type=["pt"], key="lc_model_uploader")
            if new_model_file:
                with st.expander("⚠️ Are you sure you want to overwrite the current model file?"):
                    if st.button("Confirm and Replace Light Curve Model"):
                        with open(os.path.join(artifacts_path, "best_exoplanet_model_V2.pt"), "wb") as f:
                            f.write(new_model_file.getbuffer())
                        st.success("Light Curve model updated. Clearing cache and refreshing app...")
                        st.cache_resource.clear()
                        time.sleep(2)
                        st.rerun()

        with st.expander("🔄 Reset to Original Light Curve Model"):
            st.warning("This will restore the original model file from your backup.")
            with st.expander("⚠️ This action cannot be undone. Are you absolutely sure?"):
                if st.button("Confirm and Reset Model"):
                    try:
                        artifacts_path = "Light_Curve_Models_Artifacts"
                        source_path = os.path.join(artifacts_path, "best_exoplanet_model_V3_original.pt")
                        dest_path = os.path.join(artifacts_path, "best_exoplanet_model_v4.pt")
                        shutil.copyfile(source_path, dest_path)
                        st.success("Light Curve model has been reset. Clearing cache and refreshing...")
                        st.cache_resource.clear()
                        time.sleep(2)
                        st.rerun()
                    except FileNotFoundError:
                        st.error("Original backup file ('best_exoplanet_model_V2_original.pt') not found.")

        if st.button("Lock Admin Mode"):
            st.session_state.authenticated = False
            st.rerun()



