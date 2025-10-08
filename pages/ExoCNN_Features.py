import streamlit as st
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="ExoCNN Features",
    page_icon="✨",
    layout="wide"
)

# --- Function to encode image to base64 ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- Function to set background ---
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
    st.sidebar.warning("Background image not found.")

# --- Title ---
st.title("🚀 Project Features & Capabilities")
st.markdown("---")

# --- Animated Feature Cards ---
# This single HTML block contains all the styles and structure for the feature cards.
feature_cards_html = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap');

    .feature-card {
        background: rgba(10, 20, 40, 0.6);
        border: 1px solid #2a3e5f;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 1s ease-out forwards;
        backdrop-filter: blur(5px);
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .feature-card:nth-child(2) { animation-delay: 0.2s; }
    .feature-card:nth-child(3) { animation-delay: 0.4s; }
    .feature-card:nth-child(4) { animation-delay: 0.6s; }

    .feature-icon {
        flex: 0 0 120px;
        height: 120px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .feature-description {
        flex: 1;
        color: #E0E0E0;
        font-family: 'Roboto Mono', monospace;
    }

    .feature-description h2 {
        font-family: 'Orbitron', sans-serif;
        color: #87CEEB;
        margin-top: 0;
    }

    /* --- Custom Animations --- */

    /* Home Page Animation */
    .home-anim-icon {
        font-size: 70px;
        animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); text-shadow: 0 0 5px #87CEEB; }
        70% { transform: scale(1.1); text-shadow: 0 0 25px #87CEEB; }
        100% { transform: scale(0.95); text-shadow: 0 0 5px #87CEEB; }
    }

    /* CNN Animation */
    .cnn-anim-icon {
        width: 100px;
        height: 50px;
        position: relative;
    }
    .cnn-layer {
        position: absolute;
        width: 15px;
        height: 50px;
        background: #4a90e2;
        border: 1px solid #fff;
        border-radius: 3px;
        animation: cnn-process 3s infinite;
    }
    .cnn-layer:nth-child(1) { left: 0; }
    .cnn-layer:nth-child(2) { left: 30px; animation-delay: 0.5s; }
    .cnn-layer:nth-child(3) { left: 60px; animation-delay: 1.0s; }
    @keyframes cnn-process {
        0%, 100% { background: #4a90e2; }
        50% { background: #FFD700; box-shadow: 0 0 15px #FFD700; }
    }

    /* XGBoost Animation */
    .xgb-anim-icon {
        width: 80px;
        height: 80px;
        border: 2px solid #5ba0f2;
        border-radius: 50%;
        position: relative;
    }
    .xgb-dot {
        width: 10px;
        height: 10px;
        background: #E6E6FA;
        border-radius: 50%;
        position: absolute;
        transform-origin: 40px 40px;
        animation: xgb-orbit 4s linear infinite;
    }
    .xgb-dot:nth-child(1) { animation-delay: 0s; }
    .xgb-dot:nth-child(2) { animation: xgb-orbit-2 3s linear infinite; background: #FFD700; }
    @keyframes xgb-orbit { from { transform: rotate(0deg) translateX(35px); } to { transform: rotate(360deg) translateX(35px); } }
    @keyframes xgb-orbit-2 { from { transform: rotate(0deg) translateX(20px); } to { transform: rotate(-360deg) translateX(20px); } }

    /* FITS Viewer Animation */
    .fits-anim-icon {
        width: 90px;
        height: 60px;
        border: 1px solid #8b949e;
        background: #0d1117;
        position: relative;
        overflow: hidden;
    }
    .fits-scanline {
        width: 100%;
        height: 3px;
        background: #58a6ff;
        box-shadow: 0 0 10px #58a6ff;
        position: absolute;
        top: 0;
        animation: scan 3s linear infinite;
    }
    @keyframes scan { from { top: 0; } to { top: 100%; } }

</style>

<div class="feature-card">
    <div class="feature-icon">
        <div class="home-anim-icon">🌌</div>
    </div>
    <div class="feature-description">
        <h2>Home Page: Mission Control</h2>
        <p>The central hub of the EXOHUNTERS suite. It provides a professional welcome, real-time observatory data, and a conceptual overview of the project's architecture.</p>
        <ul>
            <li><b>Live Data Feed:</b> Displays a live count of confirmed exoplanets and precise astronomical time (UTC and Julian Date).</li>
            <li><b>Visual Explanations:</b> Features an interactive transit method animation and a high-level software architecture flowchart.</li>
            <li><b>Advanced Styling:</b> Utilizes an animated particle background and futuristic fonts for an immersive user experience.</li>
        </ul>
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">
        <div class="cnn-anim-icon">
            <div class="cnn-layer"></div>
            <div class="cnn-layer"></div>
            <div class="cnn-layer"></div>
        </div>
    </div>
    <div class="feature-description">
        <h2>Light Curve Analysis: The Deep Scan</h2>
        <p>This tool employs a Convolutional Neural Network (CNN) to analyze time-series data from FITS files, detecting the subtle dips in brightness that signify an exoplanet transit.</p>
        <ul>
            <li><b>AI-Powered Detection:</b> Leverages a custom-trained PyTorch CNN model (`ExoCNN`) for high-accuracy predictions.</li>
            <li><b>Automated Preprocessing:</b> Intelligently finds, cleans, normalizes, and resamples flux data from uploaded FITS files.</li>
            <li><b>Rich Visualizations:</b> Automatically generates folded light curves, periodograms, and raw/processed plots to support the model's verdict.</li>
        </ul>
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">
        <div class="xgb-anim-icon">
            <div class="xgb-dot"></div>
            <div class="xgb-dot"></div>
        </div>
    </div>
    <div class="feature-description">
        <h2>Tabular Model Analysis: The Classifier</h2>
        <p>Utilizes a powerful XGBoost model to classify exoplanet candidates based on a set of stellar and orbital parameters. It supports both single-target prediction and batch analysis from CSV files.</p>
        <ul>
            <li><b>High-Performance Model:</b> Built on XGBoost for robust and accurate classification of planetary candidates versus false positives.</li>
            <li><b>Dual-Mode Input:</b> Accepts single-entry manual inputs or batch processing of large CSV files from missions like KOI, K2, and TESS.</li>
            <li><b>Model Interpretability:</b> Integrates SHAP (SHapley Additive exPlanations) to provide clear, visual explanations for each prediction.</li>
        </ul>
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">
        <div class="fits-anim-icon">
            <div class="fits-scanline"></div>
        </div>
    </div>
    <div class="feature-description">
        <h2>FITS Viewer: The Inspector</h2>
        <p>A comprehensive utility tool for inspecting the raw contents of any FITS (Flexible Image Transport System) file, the standard data format in astronomy.</p>
        <ul>
            <li><b>Full HDU Inspection:</b> Allows users to select and view data from any Header Data Unit (HDU) within the file.</li>
            - <li><b>Data Rendering:</b> Displays headers, tabular data, and image data with options for color mapping and scaling (Linear, Log).</li>
            - <li><b>Automated Plotting:</b> Automatically detects time-series data and generates interactive light curve plots directly from the FITS table.</li>
        </ul>
    </div>
</div>

"""


st.markdown(feature_cards_html, unsafe_allow_html=True)
