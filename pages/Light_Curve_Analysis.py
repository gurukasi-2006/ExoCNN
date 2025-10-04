import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
import base64
import plotly.express as px
import os
import tempfile
from astropy.io import fits
import shutil
import time

# --- Page Config and Background ---
st.set_page_config(layout="wide")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>.stApp {{ background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), url("data:image/jpeg;base64,{bin_str}"); background-size: cover; }}</style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
try:
    set_png_as_page_bg('background.jpg')
except Exception:
    st.sidebar.warning("background.jpg not found.")

# --- PYTORCH MODEL DEFINITION ---
class ExoCNN(nn.Module):
    def __init__(self, n_points=1000):
        super(ExoCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, padding=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        dummy_input = torch.zeros(1, 1, n_points)
        x = self.pool(self.relu(self.conv2(self.relu(self.conv1(dummy_input)))))
        self.flattened_size = x.numel()
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# --- Artifact Loading ---
@st.cache_resource
def load_light_curve_model():
    try:
        artifacts_path = "Light_Curve_Models_Artifacts"
        model_path = os.path.join(artifacts_path, "best_exoplanet_model_V2.pt")
        model = ExoCNN(n_points=1000)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'best_exoplanet_model_V2.pt' not found in 'Light_Curve_Models_Artifacts' folder.")
        return None

model = load_light_curve_model()

# --- Preprocessing Function ---
# Replace your old function with this new one
def preprocess_fits_for_prediction(fits_file, n_points=1000):
    """
    Processes the uploaded FITS file directly from memory to avoid file locking errors.
    """
    try:
        # Pass the uploaded file object DIRECTLY to fits.open()
        with fits.open(fits_file) as hdul:
            flux = None
            
            # --- NEW, MORE ROBUST LOGIC ---
            # First, search ALL HDUs for the preferred flux column (PDCSAP_FLUX)
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None and hasattr(hdu, "columns"):
                    colnames = [c.upper() for c in hdu.columns.names]
                    if "PDCSAP_FLUX" in colnames:
                        flux = hdu.data["PDCSAP_FLUX"]
                        break  # Found the best option, we can stop

            # If PDCSAP_FLUX was not found anywhere, search again for the backup (SAP_FLUX)
            if flux is None:
                for hdu in hdul:
                    if hasattr(hdu, "data") and hdu.data is not None and hasattr(hdu, "columns"):
                        colnames = [c.upper() for c in hdu.columns.names]
                        if "SAP_FLUX" in colnames:
                            flux = hdu.data["SAP_FLUX"]
                            break  # Found a backup option, we can stop
        
        if flux is None: 
            st.error("Could not find PDCSAP_FLUX or SAP_FLUX in FITS file.")
            return None, None, None

        # Create LightCurve objects for plotting later
        lc_raw = lk.LightCurve(flux=flux).remove_nans()
        flux = lc_raw.flux.value # Use the NaN-removed flux
        
        if len(flux) < 100: 
            st.error("Not enough valid data points (< 100) in the FITS file.")
            return None, None, None

        # --- Preprocessing to match your training script ---
        # 1. On-the-fly standardization
        flux_normalized = (flux - np.mean(flux)) / (np.std(flux) + 1e-6)
        
        # 2. Resample to fixed length
        idx = np.linspace(0, len(flux_normalized) - 1, n_points).astype(int)
        flux_resampled = flux_normalized[idx]
        
        # 3. Convert to a PyTorch tensor
        flux_tensor = torch.tensor(flux_resampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Create a flattened lightkurve object for the plots
        lc_flat = lk.LightCurve(flux=flux_normalized)
        
        return lc_raw, lc_flat, flux_tensor

    except Exception as e:
        st.error(f"Failed to process FITS file: {e}")
        return None, None, None

# --- Main Page UI ---
st.title("📈 Light Curve Analysis")
if model is None: st.stop()

# --- NEW TOP NAVIGATION ---
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
        lc_raw, lc_flat, flux_tensor = preprocess_fits_for_prediction(uploaded_fits)
        if flux_tensor is not None:
            with st.spinner("Running prediction..."):
                with torch.no_grad():
                    final_prob = model(flux_tensor).item()
            
            verdict = "Confirmed EXO-Planet" if final_prob >= 0.5 else "False Positive"
            
            st.header("Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                if verdict == "Confirmed EXO-Planet": st.success(f"**Verdict:** {verdict}")
                else: st.error(f"**Verdict:** {verdict}")
            with col2:
                if verdict == "Confirmed EXO-Planet":
                    st.metric("Model Confidence Score", f"{final_prob:.2%}")
                    
                else:
                    fal_prob=1-final_prob
                    st.metric("Model Confidence Score", f"{fal_prob:.2%}")
            if verdict == "Confirmed EXO-Planet": st.progress(final_prob)
            else: st.progress(fal_prob)      
            

            st.header("Visualizations")
            plot_tabs = st.tabs(["Folded Light Curve", "Periodogram", "Processed Light Curve", "Raw Light Curve"])

            with plot_tabs[0]:
                st.subheader("Folded Light Curve (at highest power)")
                with st.spinner("Searching for periodic signals and folding light curve..."):
                    try:
                        # This finds the strongest signal
                        periodogram = lc_flat.to_periodogram(method='bls')
                        best_period = periodogram.period_at_max_power
                        
                        # This folds the light curve on that signal
                        folded_lc = lc_flat.fold(period=best_period)
                        
                        # This creates the plot
                        fig_folded = px.scatter(x=folded_lc.time.value, y=folded_lc.flux.value, title=f"Light Curve Folded at Best Period ({best_period.value:.4f} days)")
                        st.plotly_chart(fig_folded, use_container_width=True)
                        st.success(f"The strongest periodic signal was found at {best_period.value:.4f} days.")
                    
                    except Exception as e:
                        # This warning will appear if the above fails
                        st.warning(f"Could not generate a folded light curve. This usually means no strong periodic signal was found in the data.")

            with plot_tabs[1]:
                st.subheader("Periodogram (Signal Search)")
                with st.spinner("Generating periodogram..."):
                    try:
                        periodogram = lc_flat.to_periodogram(method='bls')
                        fig_periodogram = px.line(x=periodogram.period.value, y=periodogram.power.value, title="Box-Least-Squares (BLS) Periodogram")
                        st.plotly_chart(fig_periodogram, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate a periodogram: {e}")
           
            with plot_tabs[2]:
                st.subheader("Processed (Normalized) Light Curve")
                with st.spinner("Generating processed light curve..."):
                    fig_processed = px.line(x=lc_flat.time.value, y=lc_flat.flux.value, title="Normalized Light Curve")
                    st.plotly_chart(fig_processed, use_container_width=True)

            with plot_tabs[3]:
                st.subheader("Raw Light Curve")
                with st.spinner("Generating raw light curve..."):
                    fig_raw = px.line(x=lc_raw.time.value, y=lc_raw.flux.value, title="Raw Light Curve from FITS File")
                    st.plotly_chart(fig_raw, use_container_width=True)

# --- ADMIN VIEW ---
elif page == "⚙️ Admin & Model Management":
    st.header("⚙️ Admin & Model Management")
    if 'authenticated' not in st.session_state: 
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.warning("This section is restricted. Please enter the password to unlock.")
        password = st.text_input("Admin Password", type="password", key="lc_admin_password")
        if st.button("Unlock"):
            if password == "2030":
                st.session_state.authenticated = True; st.rerun()
            else:
                st.error("Incorrect password.")
    else: 
        st.success("Admin Mode Unlocked.")
        
        with st.expander("⬆️ Upload New Light Curve Model"):
            st.write("Manually replace the current PyTorch model file (`.pt`).")
            artifacts_path = "Light_Curve_Models_Artifacts"
            new_model_file = st.file_uploader("Upload best_exoplanet_model_V2.pt", type=["pt"], key="lc_model_uploader")
            if new_model_file:
                with st.expander("⚠️ Are you sure you want to overwrite the current model file?"):
                    if st.button("Confirm and Replace Light Curve Model"):
                        with open(os.path.join(artifacts_path, "best_exoplanet_model_V2.pt"), "wb") as f:
                            f.write(new_model_file.getbuffer())
                        st.success("Light Curve model updated. Clearing cache and refreshing app...")
                        st.cache_resource.clear()
                        time.sleep(2); st.rerun()

        with st.expander("🔄 Reset to Original Light Curve Model"):
            st.warning("This will restore the original model file from your backup.")
            with st.expander("⚠️ This action cannot be undone. Are you absolutely sure?"):
                if st.button("Confirm and Reset Model"):
                    try:
                        artifacts_path = "Light_Curve_Models_Artifacts"
                        source_path = os.path.join(artifacts_path, "best_exoplanet_model_V2_original.pt")
                        dest_path = os.path.join(artifacts_path, "best_exoplanet_model_V2.pt")
                        shutil.copyfile(source_path, dest_path)
                        st.success("Light Curve model has been reset. Clearing cache and refreshing...")
                        st.cache_resource.clear()
                        time.sleep(2); st.rerun()
                    except FileNotFoundError:
                        st.error("Original backup file ('best_exoplanet_model_V2_original.pt') not found.")
        
        if st.button("Lock Admin Mode"):
            st.session_state.authenticated = False
            st.rerun()

            
