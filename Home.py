#Importing Libraries
import streamlit as st
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import time
from astropy.time import Time

# --- Page Configuration ---
st.set_page_config(
    page_title="EXOHUNTERS",
    page_icon="🪐",
    layout="wide"
)

# --- Enhanced Styling and Fonts ---
def load_custom_styling():
    """Injects custom CSS for fonts, animations, and background."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap');

        /* --- General Background & App Styling --- */
        .stApp {
            background: linear-gradient(180deg, #000428 0%, #000000 70%);
        }

        /* --- Title and Header Fonts & Animations --- */
        h1 {
            font-family: 'Orbitron', sans-serif;
            color: #87CEEB;
            text-shadow: 0 0 15px #87CEEB;
            animation: fadeIn 2s ease-in-out;
        }

        h2, h3 { /* For st.header and st.subheader */
            font-family: 'Orbitron', sans-serif;
            color: #E6E6FA;
            animation: fadeIn 2.5s ease-in-out;
        }

        /* --- Styling for Live Data (Clocks and Counter) --- */
        .live-clock {
            font-family: 'Roboto Mono', monospace;
            color: #E6E6FA;
            font-size: 8.2rem;
            font-weight: 700;
            text-shadow: 0 0 8px #ADD8E6;
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: color 0.5s ease-in-out, text-shadow 0.5s ease-in-out;
        }

        .live-clock:hover {
            color: #FFFFFF;
            text-shadow: 0 0 12px #FFFFFF;
        }

        .clock-label {
            font-family: 'Roboto Mono', monospace;
            font-size: 4rem;
            color: #87CEEB;
            text-align: center;
        }

        /* --- Keyframe Animations --- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
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

st.title("🌌 EXOHUNTERS")
st.subheader("An AI-Powered Exoplanet Analysis Suite")

st.markdown("---")

st.header("Live Observatory Data")
col1, col2, col3 = st.columns(3)

# --- UPDATED: All three dashboard items now use the same animated style ---
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

    ### Available Tools:
    - **Light Curve Analysis**: Upload a FITS file and use a deep learning (CNN) model to detect transit signals.
    - **Tabular Model Analysis**: Input stellar and transit parameters or CSV file to get a prediction from a powerful XGBoost model.
    - **FITS Viewer**: A utility to inspect the contents of any FITS file.
    - **Admin Tools**: A protected area for managing the AI models.

    -Made by-
        Mugeshkumar S,
        GuruKasi M
    """
)

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

