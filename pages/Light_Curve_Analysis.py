import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
import base64
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from scipy.fft import fft
from astropy.io import fits
import time
import shutil
from styling import add_advanced_loading_animation, load_custom_styling_back


add_advanced_loading_animation()
load_custom_styling_back()
#Page Config & Styling
st.set_page_config(layout="wide")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
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
    set_png_as_page_bg('exo_assets/background.jpg')
except Exception:
    pass

#PYTORCH MODEL
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

# XAI Functions
def compute_integrated_gradients(model, input_tensor, baseline=None, steps=50):
    """
    Compute Integrated Gradients for input attribution.
    """
    # Ensure input is 3D: [batch, channels, sequence_length]
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.squeeze(0)  # Remove extra dimension
    
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    # Generate interpolated inputs between baseline and actual input
    alphas = torch.linspace(0, 1, steps)
    
    # Compute gradients
    gradients = []
    for alpha in alphas:
        interpolated_input = baseline + alpha * (input_tensor - baseline)
        interpolated_input.requires_grad = True
        
        output = model(interpolated_input)
        model.zero_grad()
        output.backward()
        
        gradients.append(interpolated_input.grad.detach().clone())
    
    # Average gradients and multiply by input difference
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (input_tensor - baseline) * avg_gradients
    
    return integrated_grads.squeeze().cpu().numpy()

def compute_gradient_saliency(model, input_tensor):
    """
    Compute simple gradient-based saliency map.
    """
    # Ensure input is 3D: [batch, channels, sequence_length]
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.squeeze(0)
    
    input_tensor = input_tensor.clone()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    
    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    return saliency

def compute_smoothgrad(model, input_tensor, noise_level=0.1, n_samples=50):
    """
    Compute SmoothGrad by averaging gradients of noisy inputs.
    """
    # Ensure input is 3D: [batch, channels, sequence_length]
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.squeeze(0)
    
    saliency_maps = []
    
    for _ in range(n_samples):
        # Add random noise
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        noisy_input = noisy_input.clone()
        noisy_input.requires_grad = True
        
        # Compute gradient
        output = model(noisy_input)
        model.zero_grad()
        output.backward()
        
        saliency_maps.append(noisy_input.grad.abs().cpu().numpy())
    
    # Average the saliency maps
    smoothgrad = np.mean(saliency_maps, axis=0).squeeze()
    return smoothgrad

def get_feature_importance_scores(attributions, n_flux_points=1000):
    """
    Compute aggregated importance scores for flux and FFT features.
    """
    flux_importance = np.abs(attributions[:n_flux_points]).sum()
    fft_importance = np.abs(attributions[n_flux_points:]).sum()
    
    total = flux_importance + fft_importance
    flux_pct = (flux_importance / total * 100) if total > 0 else 0
    fft_pct = (fft_importance / total * 100) if total > 0 else 0
    
    return flux_pct, fft_pct

def create_attribution_plot(attributions, flux_normalized, n_flux_points=1000):
    """
    Create an interactive plot showing the light curve with attribution overlay.
    """
    # Extract flux attributions
    flux_attributions = attributions[:n_flux_points]
    
    # Normalize attributions for visualization
    attr_normalized = (flux_attributions - flux_attributions.min()) / (flux_attributions.max() - flux_attributions.min() + 1e-8)
    
    # Create figure with dual y-axes
    fig = go.Figure()
    
    # Add flux line
    fig.add_trace(go.Scatter(
        y=flux_normalized,
        mode='lines',
        name='Normalized Flux',
        line=dict(color='lightblue', width=2),
        yaxis='y1'
    ))
    
    # Add attribution heatmap overlay
    fig.add_trace(go.Scatter(
        y=attr_normalized,
        mode='lines',
        name='Feature Importance',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Light Curve with Feature Attribution Overlay',
        xaxis=dict(title='Time Step'),
        yaxis=dict(title='Normalized Flux', side='left', color='lightblue'),
        yaxis2=dict(title='Attribution Score', overlaying='y', side='right', color='red'),
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_fft_attribution_plot(attributions, n_flux_points=1000):
    """
    Create a plot showing FFT feature attributions.
    """
    fft_attributions = attributions[n_flux_points:]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=np.abs(fft_attributions),
        name='FFT Feature Importance',
        marker=dict(color='purple')
    ))
    
    fig.update_layout(
        title='FFT Feature Importance',
        xaxis=dict(title='Frequency Bin'),
        yaxis=dict(title='Absolute Attribution'),
        height=400
    )
    
    return fig

# Artifact Loading
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

# Preprocessing Function
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
            return None, None, None, None

        lc_raw = lk.LightCurve(flux=flux).remove_nans()
        flux = lc_raw.flux.value

        if len(flux) < 100:
            st.error("Not enough valid data points (< 100) in the FITS file.")
            return None, None, None, None

        # CORRECTED PREPROCESSING LOGIC
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
        
        return lc_raw, lc_flat, final_tensor, flux_normalized

    except Exception as e:
        st.error(f"Failed to process FITS file: {e}")
        return None, None, None, None

#Main Page UI
st.title("📈 Light Curve Analysis with XAI")
if model is None or scaler is None:
    st.stop()

page = st.radio(
    "Choose a view:",
    ["Analysis", "⚙️ Admin & Model Management"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ANALYSIS VIEW
if page == "Analysis":
    st.write("Upload a FITS file to analyze it with the `ExoCNN` deep learning model.")
    uploaded_fits = st.file_uploader("Upload or Drag and Drop your FITS file here", type=["fits", "fit"])

    if uploaded_fits is not None:
        lc_raw, lc_flat, flux_tensor, flux_normalized = preprocess_fits_for_prediction(uploaded_fits, n_points=N_POINTS_BASE)
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
                    # Fix endianness issue
                    time_data = np.asarray(lc_raw.time.value, dtype=np.float64)
                    flux_data = np.asarray(lc_raw.flux.value, dtype=np.float64)
                    fig_raw = px.line(x=time_data, y=flux_data, title="Raw Light Curve from FITS File")
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

            # XAI Section
            st.header("🔍 Explainability Analysis (XAI)")
            st.write("Understanding why the model made this prediction:")
            
            xai_method = st.selectbox(
                "Select XAI Method:",
                ["Integrated Gradients", "Gradient Saliency", "SmoothGrad"],
                help="Different methods for explaining model predictions"
            )
            
            with st.spinner(f"Computing {xai_method}..."):
                if xai_method == "Integrated Gradients":
                    attributions = compute_integrated_gradients(model, flux_tensor.clone())
                elif xai_method == "Gradient Saliency":
                    attributions = compute_gradient_saliency(model, flux_tensor.clone())
                else:  # SmoothGrad
                    attributions = compute_smoothgrad(model, flux_tensor.clone())
                
                # Compute feature importance
                flux_pct, fft_pct = get_feature_importance_scores(attributions, N_POINTS_BASE)
                
                # Display feature importance
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Time Series Contribution", f"{flux_pct:.1f}%")
                with col2:
                    st.metric("Frequency Domain Contribution", f"{fft_pct:.1f}%")
                
                # Create attribution visualizations
                st.subheader("Feature Attribution Visualization")
                fig_attr = create_attribution_plot(attributions, flux_normalized, N_POINTS_BASE)
                st.plotly_chart(fig_attr, use_container_width=True)
                
                st.info("🔴 **Red shaded areas** indicate regions of the light curve that had the highest influence on the model's decision.")
                
                # FFT attribution plot
                with st.expander("📊 View Frequency Domain Attribution"):
                    fig_fft_attr = create_fft_attribution_plot(attributions, N_POINTS_BASE)
                    st.plotly_chart(fig_fft_attr, use_container_width=True)
                    st.info("This shows which frequency components were most important for the prediction.")

            

#  ADMIN VIEW
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
            st.write("Manually replace the current PyTorch model file (`best_exoplanet_model_V4.pt`).")
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
                        source_path = os.path.join(artifacts_path, "best_exoplanet_model_v4_original.pt")
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
