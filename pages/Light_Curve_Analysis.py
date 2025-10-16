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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime
import plotly.io as pio


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

def generate_pdf_report(filename, verdict, confidence, final_prob, flux_pct, fft_pct, 
                       xai_method, best_period=None, figures_dict=None):
    """
    Generate a comprehensive PDF report of the light curve analysis.
    Optimized for faster generation with reduced image quality for speed.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Exoplanet Light Curve Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    metadata_style = styles['Normal']
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
    story.append(Paragraph(f"<b>Analyzed File:</b> {filename}", metadata_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    # Results table
    verdict_color = '#28a745' if verdict == "Confirmed EXO-Planet" else '#dc3545'
    results_data = [
        ['Analysis Result', 'Value'],
        ['Verdict', verdict],
        ['Model Confidence', f'{confidence:.2%}'],
        ['Raw Prediction Score', f'{final_prob:.4f}'],
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # XAI Analysis
    story.append(Paragraph("Explainability Analysis (XAI)", heading_style))
    story.append(Paragraph(f"<b>Method Used:</b> {xai_method}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    xai_data = [
        ['Feature Type', 'Contribution'],
        ['Time Series Features', f'{flux_pct:.1f}%'],
        ['Frequency Domain Features', f'{fft_pct:.1f}%'],
    ]
    
    xai_table = Table(xai_data, colWidths=[3*inch, 3*inch])
    xai_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 0.2*inch))
    
    if best_period:
        story.append(Paragraph(f"<b>Best Period Detected:</b> {best_period:.4f} days", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Add page break before visualizations
    story.append(PageBreak())
    
    # Visualizations section
    story.append(Paragraph("Visualizations", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    if figures_dict:
        for title, fig in figures_dict.items():
            story.append(Paragraph(title, styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Convert plotly figure to image with optimized settings for speed
            # Reduced resolution and quality for faster generation
            img_bytes = pio.to_image(fig, format='png', width=600, height=350, scale=1)
            img_buffer = BytesIO(img_bytes)
            img = Image(img_buffer, width=5.5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(PageBreak())
    story.append(Paragraph("Technical Notes", heading_style))
    notes_text = """
    This report was generated using the ExoCNN deep learning model for exoplanet detection.
    The model analyzes both time-series flux data and frequency domain features (FFT) to make predictions.
    Explainability methods (XAI) provide insights into which features influenced the model's decision.
    """
    story.append(Paragraph(notes_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<i>Generated by Light Curve Analysis System</i>", 
                          ParagraphStyle('Footer', parent=styles['Normal'], 
                                       fontSize=9, textColor=colors.grey, alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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

            # PDF Report Generation Section - OPTIMIZED
            st.markdown("---")
            st.header("📄 Download Analysis Report")
            
            # Initialize generation state
            if 'pdf_generating' not in st.session_state:
                st.session_state['pdf_generating'] = False
            
            # Create columns for better layout
            col_btn1, col_btn2 = st.columns([1, 3])
            
            with col_btn1:
                generate_clicked = st.button("📥 Generate PDF Report", type="primary", key="gen_pdf_btn")
            
            if generate_clicked:
                st.session_state['pdf_generating'] = True
                st.session_state['pdf_buffer'] = None  # Clear old PDF
            
            # Show loading immediately when button is clicked
            if st.session_state['pdf_generating']:
                # Create placeholder for progress
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0, text="🔄 Initializing PDF generation...")
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Prepare data
                        status_text.info("📊 Step 1/3: Collecting figures and data...")
                        progress_bar.progress(10, text="📊 Collecting figures...")
                        
                        # Collect all figures
                        figures_dict = {
                            "Raw Light Curve": fig_raw,
                            "Processed Light Curve": fig_processed,
                            "Feature Attribution": fig_attr,
                        }
                        
                        progress_bar.progress(20, text="📊 Processing periodogram...")
                        
                        # Add periodogram if available
                        try:
                            periodogram = lc_flat.to_periodogram(method='bls')
                            fig_periodogram = px.line(
                                x=periodogram.period.value, 
                                y=periodogram.power.value, 
                                title="Box-Least-Squares (BLS) Periodogram"
                            )
                            figures_dict["Periodogram"] = fig_periodogram
                        except:
                            pass
                        
                        progress_bar.progress(30, text="📊 Processing folded light curve...")
                        
                        # Add folded light curve if available
                        best_period_val = None
                        try:
                            periodogram = lc_flat.to_periodogram(method='bls')
                            best_period_val = periodogram.period_at_max_power.value
                            folded_lc = lc_flat.fold(period=periodogram.period_at_max_power)
                            fig_folded = px.scatter(
                                x=folded_lc.time.value, 
                                y=folded_lc.flux.value, 
                                title=f"Light Curve Folded at Best Period"
                            )
                            figures_dict["Folded Light Curve"] = fig_folded
                        except:
                            pass
                        
                        # Step 2: Convert visualizations
                        status_text.info("🖼️ Step 2/3: Converting visualizations to images...")
                        progress_bar.progress(50, text="🖼️ Converting visualizations...")
                        
                        # Step 3: Build PDF
                        status_text.info("📝 Step 3/3: Building PDF document...")
                        progress_bar.progress(70, text="📝 Building PDF document...")
                        
                        pdf_buffer = generate_pdf_report(
                            filename=uploaded_fits.name,
                            verdict=verdict,
                            confidence=confidence,
                            final_prob=final_prob,
                            flux_pct=flux_pct,
                            fft_pct=fft_pct,
                            xai_method=xai_method,
                            best_period=best_period_val,
                            figures_dict=figures_dict
                        )
                        
                        # Finalize
                        progress_bar.progress(100, text="✅ PDF generation complete!")
                        status_text.success("✅ PDF Report generated successfully! Click below to download.")
                        
                        # Store PDF in session state
                        st.session_state['pdf_buffer'] = pdf_buffer
                        st.session_state['pdf_generating'] = False
                        
                        time.sleep(1)  # Brief pause to show success
                        st.rerun()  # Refresh to show download button
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.error(f"❌ PDF generation failed: {e}")
                        st.info("💡 Note: Ensure required packages are installed: `pip install reportlab kaleido`")
                        st.session_state['pdf_generating'] = False
            
            # Show download button if PDF is ready (only when not generating)
            if 'pdf_buffer' in st.session_state and st.session_state['pdf_buffer'] is not None and not st.session_state.get('pdf_generating', False):
                st.success("✅ Your PDF report is ready!")
                
                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=st.session_state['pdf_buffer'],
                        file_name=f"exoplanet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="secondary",
                        on_click=lambda: st.session_state.update({'pdf_downloaded': True})
                    )
                
                # Option to generate a new report
                with col_dl2:
                    if st.button("🔄 Generate New Report", key="new_report_btn"):
                        st.session_state['pdf_buffer'] = None
                        st.session_state['pdf_generating'] = False
                        st.rerun()

            

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
