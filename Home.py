import streamlit as st
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import time
from astropy.time import Time
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="EXOHUNTERS",
    page_icon="🪐",
    layout="wide"
)

# --- Function to encode image to base64 ---
def get_image_as_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- Enhanced Styling and Fonts (ALL CSS IS NOW CONSOLIDATED HERE) ---
def load_custom_styling():
    """Injects all custom CSS for the entire page in one block."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap');

        /* --- General Background & App Styling --- */
        .stApp {{
            background: linear-gradient(180deg, #000428 0%, #000000 70%);
        }}

        /* --- Title and Header Fonts & Animations --- */
        .custom-title h1 {{
            font-family: 'Orbitron', sans-serif;
            color: #87CEEB;
            text-shadow: 0 0 15px #87CEEB;
            animation: fadeIn 2s ease-in-out;
            display: inline-block;
            vertical-align: middle;
        }}

        h2, h3 {{ /* For st.header and st.subheader */
            font-family: 'Orbitron', sans-serif;
            color: #E6E6FA;
            animation: fadeIn 2.5s ease-in-out;
        }}

        /* --- Styling for Live Data (Clocks and Counter) - FONT SIZE CORRECTED --- */
        .live-clock {{
            font-family: 'Roboto Mono', monospace;
            color: #E6E6FA;
            font-size: 8.2rem;  /* <-- CORRECTED LARGE FONT SIZE */
            font-weight: 700;
            text-shadow: 0 0 8px #ADD8E6;
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: color 0.5s ease-in-out, text-shadow 0.5s ease-in-out;
        }}
        .live-clock:hover {{
            color: #FFFFFF;
            text-shadow: 0 0 12px #FFFFFF;
        }}
        .clock-label {{
            font-family: 'Roboto Mono', monospace;
            font-size: 4rem;  /* <-- CORRECTED LARGE FONT SIZE */
            color: #87CEEB;
            text-align: center;
        }}
        
        /* --- Logo Styling and Animation --- */
        .logo-container {{
            display: inline-block;
            vertical-align: middle;
            margin-left: 20px;
            transform: scale(0);
            animation: logoPopIn 1s ease-out forwards;
            animation-delay: 0.8s;
        }}
        .logo-container img {{
            height: 120px;
            width: 120px;
            border-radius: 50%;
            box-shadow: 0 0 30px rgba(135, 206, 235, 0.8);
        }}

        /* --- Keyframe Animations --- */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes logoPopIn {{
            0% {{ transform: scale(0) rotate(-180deg); opacity: 0; }}
            60% {{ transform: scale(1.1) rotate(10deg); opacity: 1; }}
            100% {{ transform: scale(1) rotate(0deg); opacity: 1; }}
        }}

        /* --- Flowchart Styles --- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .flowchart-container {{
            font-family: 'Roboto Mono', monospace;
            color: #E0E0E0;
            text-align: center;
            padding: 2rem 1rem;
            background-color: rgba(10, 20, 40, 0.5);
            border: 1px solid #2a3e5f;
            border-radius: 15px;
            margin-top: 2rem;
            overflow: hidden;
        }}
        .flowchart-layer {{
            margin: 2.5rem 0;
            opacity: 0;
            animation: fadeInUp 1s ease-out forwards;
        }}
        .flowchart-node {{
            display: inline-block;
            background-color: #1a253c;
            border: 1px solid #4a90e2;
            border-radius: 8px;
            padding: 12px 20px;
            margin: 0 15px;
            box-shadow: 0 0 15px rgba(74, 144, 226, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .flowchart-node:hover {{
            transform: translateY(-5px);
            box-shadow: 0 0 25px rgba(74, 144, 226, 0.6);
        }}
        .ai-core {{
            background-color: #4a90e2;
            color: #FFFFFF;
            font-weight: 700;
            padding: 15px 25px;
        }}
        .flowchart-arrow {{
            margin: 1rem 0;
            opacity: 0;
            animation: fadeIn 1s ease-out forwards; 
            font-size: 1.5rem;
        }}
        #layer-1 {{ animation-delay: 0.5s; }}
        #arrow-1 {{ animation-delay: 1.5s; }}
        #layer-2 {{ animation-delay: 2.0s; }}
        #arrow-2 {{ animation-delay: 3.0s; }}
        #layer-3 {{ animation-delay: 3.5s; }}
        #arrow-3 {{ animation-delay: 4.5s; }}
        #layer-4 {{ animation-delay: 5.0s; }}
    </style>
    """, unsafe_allow_html=True)


# --- tsParticles Animated Background ---
def animated_background():
    """Injects HTML/CSS/JS for a full-screen, animated starfield."""
    st.components.v1.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        #tsparticles {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
        }
    </style>
    </head>
    <body>
        <div id="tsparticles"></div>
        <script src="https://cdn.jsdelivr.net/npm/tsparticles@2.12.0/tsparticles.bundle.min.js"></script>
        <script>
            tsParticles.load("tsparticles", {
                "fullScreen": { "enable": true, "zIndex": -1 },
                "particles": {
                    "number": { "value": 200, "density": { "enable": true, "value_area": 800 }},
                    "color": { "value": ["#FFFFFF", "#E6E6FA", "#ADD8E6"] },
                    "shape": { "type": "circle" },
                    "opacity": {
                        "value": 0.8,
                        "random": true,
                        "anim": { "enable": true, "speed": 0.6, "opacity_min": 0.1, "sync": false }
                    },
                    "size": {
                        "value": 2.5,
                        "random": true,
                        "anim": { "enable": true, "speed": 1.5, "size_min": 0.5, "sync": false }
                    },
                    "line_linked": { "enable": false },
                    "move": {
                        "enable": true,
                        "speed": 0.4,
                        "direction": "none",
                        "random": true,
                        "straight": false,
                        "out_mode": "out"
                    }
                },
                "interactivity": {
                    "detect_on": "canvas",
                    "events": { "onhover": { "enable": true, "mode": "bubble" }, "onclick": { "enable": false }},
                    "modes": { "bubble": { "distance": 200, "size": 4, "duration": 2, "opacity": 1}}
                },
                "retina_detect": true
            });
        </script>
    </body>
    </html>
    """, height=0)


# --- Live Data Fetching Functions (Unchanged) ---
@st.cache_data(ttl=3600)
def get_live_exoplanet_count():
    try:
        count = NasaExoplanetArchive.query_criteria(table="ps", select="count(pl_name)")[0][0]
        return count
    except Exception:
        return 5641

def get_astro_time():
    """Calculates current UTC and Julian Date."""
    now = Time.now()
    utc_time = now.utc.iso.split('.')[0].replace("T", " ")
    julian_date = f"{now.jd:.5f}"
    return utc_time, julian_date

# --- Main Page UI ---
load_custom_styling()
animated_background()

# --- Logo and Title Display ---
logo_path = "exohunters_logo.png"
logo_base64 = get_image_as_base64(logo_path)

if logo_base64:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
        <div class="custom-title"><h1>🌌 EXOHUNTERS</h1></div>
        <div class="logo-container">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Exohunters Logo">
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🌌 EXOHUNTERS")
    st.warning("Logo image not found at 'exohunters_logo.png'. Please check the file path.")

st.header("An AI-Powered Exoplanet Analysis Suite")

st.markdown("---")

st.header("Live Observatory Data")
col1, col2, col3 = st.columns(3)

with col1:
    live_count = get_live_exoplanet_count()
    st.markdown(f"""
    <p class="clock-label">Confirmed Exoplanets</p>
    <p class="live-clock">{live_count:,}</p>
    """, unsafe_allow_html=True)

with col2:
    utc_placeholder = st.empty()

with col3:
    jd_placeholder = st.empty()

st.markdown("---")

st.header("About This Application")
st.markdown(
    """
    This is a professional interface for a suite of AI models designed to detect
    exoplanets from astrophysical data. Use the navigation sidebar to access the analysis tools.

    ### Period (The Rhythm):
    - **What it is:** How often the light dims.
    - **Why it matters:** A real planet has a perfect, repeating rhythm. If the light dims exactly every 3 days, it's a good sign. If the timing is random, it's not a planet.
    ### Duration (The Crossing Time):
    - **What it is:** How long the light stays dim each time.
    - **Why it matters:** The time it takes for the bug to cross the light bulb tells us how fast it's moving. A dip that's too quick is likely just a glitch.
    ### Depth (The Dimness):
    - **What it is:** How much the light dims.
    - **Why it matters:** This tells you the size of the object. A tiny bug (a planet) will only make the light a tiny bit dimmer. If the light gets very dim, the object was probably another massive light bulb (another star), not a planet.
    ### Signal-to-Noise Ratio / SNR (The Clarity):
    - **What it is:** How clear and obvious the dimming is.
    - **Why it matters:** A sharp, clear dip is a believable signal (High SNR). A fuzzy, faint dip that's hard to see against the background flicker is probably not real (Low SNR).
    ### Stellar Parameters (Info about the Star):
    - **What they are:** Knowing details about the star itself (Is it big? Is it hot?).
    - **Why they matter:**  If you know you're looking at a small star, but you see a shadow that suggests a giant planet, something is wrong. Knowing about the star helps the AI check if the story adds up.

    ### Available Tools:
    - **Light Curve Analysis**: Upload a FITS file and use a deep learning (CNN) model to detect transit signals.
    - **Tabular Model Analysis**: Input stellar and transit parameters or CSV file to get a prediction from a powerful XGBoost model.
    - **FITS Viewer**: A utility to inspect the contents of any FITS file.
    - **Admin Tools**: A protected area for managing the AI models.
    """
)

# --- ANIMATED FLOWCHART (HTML ONLY, NO STYLE TAG) ---
st.markdown("---")
st.header("Software Architecture & Data Flow")

flowchart_html = """
<div class="flowchart-container">
    <div id="layer-1" class="flowchart-layer">
        <div class="flowchart-node">☁️ Cloud Storage & APIs</div>
        <div class="flowchart-node">🗃️ FITS Files & Databases</div>
    </div>
    <div id="arrow-1" class="flowchart-arrow">⬇️</div>
    <div id="layer-2" class="flowchart-layer">
        <div class="flowchart-node">🧹 Data Preprocessing</div>
        <div class="flowchart-node ai-core">🧠 AI CORE (CNN & XGBoost Models)</div>
        <div class="flowchart-node">📈 Predictive Analytics</div>
    </div>
    <div id="arrow-2" class="flowchart-arrow">⬇️</div>
    <div id="layer-3" class="flowchart-layer">
        <div class="flowchart-node">📉 Light Curve Analysis</div>
        <div class="flowchart-node">📊 Tabular Model Analysis</div>
        <div class="flowchart-node">🔭 FITS Viewer</div>
    </div>
    <div id="arrow-3" class="flowchart-arrow">⬇️</div>
    <div id="layer-4" class="flowchart-layer">
        <div class="flowchart-node">🖥️ Professional User Interface</div>
    </div>
</div>
"""
st.markdown(flowchart_html, unsafe_allow_html=True)

# --- Live Clock Loop ---
while True:
    utc_time, julian_date = get_astro_time()
    
    utc_placeholder.markdown(f"""
    <p class="clock-label">Universal Time (UTC)</p>
    <p class="live-clock">{utc_time}</p>
    """, unsafe_allow_html=True)
    
    jd_placeholder.markdown(f"""
    <p class="clock-label">Julian Date (JD)</p>
    <p class="live-clock">{julian_date}</p>
    """, unsafe_allow_html=True)
    
    time.sleep(1)
