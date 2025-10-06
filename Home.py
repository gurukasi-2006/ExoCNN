import streamlit as st
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import time
from astropy.time import Time
import base64
import os
import streamlit.components.v1 as components
from styling import add_advanced_loading_animation


# --- GOOGLE ANALYTICS TRACKING ---
def inject_ga():
    GA_ID = "G-CK7R7RTBNK"  

    GA_SCRIPT = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{GA_ID}');
        </script>
    """
    components.html(GA_SCRIPT, height=0)

add_advanced_loading_animation()
inject_ga()
# --- Page Configuration ---
st.set_page_config(
    page_title="EXOHUNTERS",
    page_icon="🪐",
    layout="wide"
)

# --- Function to encode image to base64 ---
def get_image_as_base64(path):
    """Gets the base64 string of an image file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- Enhanced Styling and Fonts (ALL CSS IS NOW CONSOLIDATED HERE) ---
# --- Enhanced Styling and Fonts (ALL CSS IS NOW CONSOLIDATED HERE) ---
def load_custom_styling():
    """Injects all custom CSS for the entire page in one block."""
    # Encode the local sidebar background to base64
    sidebar_bg_path = "sidebar_background.jpg"
    sidebar_bg_base64 = get_image_as_base64(sidebar_bg_path)

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap');

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] > div:first-child {{
            background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("data:image/jpeg;base64,{sidebar_bg_base64}");
            background-position: center; 
            background-repeat: no-repeat;
            background-size: cover;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid #2a3e5f;
        }}
        
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

        /* --- Styling for Live Data (Clocks and Counter) --- */
        .live-clock {{
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
        }}
        .live-clock:hover {{
            color: #FFFFFF;
            text-shadow: 0 0 12px #FFFFFF;
        }}
        .clock-label {{
            font-family: 'Roboto Mono', monospace;
            font-size: 4rem;
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

        /* --- REBUILT: Transit Animation Styles --- */
        .transit-container {{
            position: relative;
            width: 100%;
            max-width: 700px;
            margin: 40px auto;
            padding: 30px;
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 15px;
            font-family: 'Roboto Mono', monospace;
            color: #c9d1d9;
        }}
        .scene {{
            position: relative;
            height: 100px;
            display: flex;
            align-items: center;
        }}
        .star {{
            width: 80px;
            height: 80px;
            background-color: #f0e68c;
            border-radius: 50%;
            box-shadow: 0 0 25px #f0e68c;
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }}
        .planet {{
            width: 15px;
            height: 15px;
            background-color: #6cb6ff;
            border-radius: 50%;
            position: absolute;
            animation: transit-pass 8s linear infinite;
        }}
        .orbit-path {{
            position: absolute;
            width: 100%;
            border-top: 1px dashed rgba(255, 255, 255, 0.2);
        }}
        .light-curve-graph {{
            position: relative;
            height: 120px;
            width: 100%;
            border-left: 1px solid #8b949e;
            border-bottom: 1px solid #8b949e;
            margin-top: 40px;
        }}
        .curve-path {{
            position: absolute;
            width: 100%;
            height: 100%;
            clip-path: path('M 0 10 L 210 10 C 245 10, 245 80, 280 80 L 420 80 C 455 80, 455 10, 490 10 L 700 10');
            overflow: hidden;
        }}
        .curve-path::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #58a6ff;
            transform: translateX(-100%);
            animation: draw-line 8s linear infinite;
        }}
        .graph-labels span {{
            position: absolute;
            color: #8b949e;
            font-size: 0.8rem;
        }}
        .label-time {{ bottom: -25px; left: 45%; }}
        .label-brightness {{ bottom: 45%; left: -60px; transform: rotate(-90deg); }}

        @keyframes transit-pass {{
            0%   {{ left: 0%; }}
            100% {{ left: 100%; }}
        }}
        @keyframes draw-line {{
            0%   {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(0%); }}
        }}
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


# --- Live Data Fetching Functions Non Live ---
@st.cache_data(ttl=3600)
def get_live_exoplanet_count():
    try:
        count = 6022
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
logo_path = "exohunters_logo.png" # Corrected path
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
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #E0E0E0; font-family: 'Roboto Mono', monospace;">
        Created by<br>
        <strong>S Mugeshkumar<br>M Gurukasi</strong>
    </div>
    """,
    unsafe_allow_html=True
)
# --- TRANSIT ANIMATION ---
st.header("The Transit Method: A Visual Explanation")
st.components.v1.html("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: transparent;
            font-family: 'Roboto Mono', monospace;
            color: #E6E6FA;
        }

        .animation-container {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .star-animation {
            flex: 1;
            min-width: 400px;
            background: rgba(10, 20, 40, 0.5);
            border: 2px solid #4a90e2;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(74, 144, 226, 0.3);
        }

        .graph-container {
            flex: 1;
            min-width: 400px;
            background: rgba(10, 20, 40, 0.5);
            border: 2px solid #4a90e2;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 25px rgba(74, 144, 226, 0.3);
        }

        canvas {
            display: block;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }

        .controls {
            text-align: center;
            margin-top: 20px;
        }

        button {
            background: #4a90e2;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
            font-family: 'Roboto Mono', monospace;
            box-shadow: 0 0 15px rgba(74, 144, 226, 0.4);
        }

        button:hover {
            background: #5ba0f2;
            box-shadow: 0 0 25px rgba(74, 144, 226, 0.6);
            transform: translateY(-2px);
        }

        button:active {
            transform: translateY(0);
        }

        .info {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background: rgba(74, 144, 226, 0.1);
            border-radius: 10px;
            border: 1px solid #4a90e2;
        }

        .label {
            text-align: center;
            font-size: 18px;
            color: #87CEEB;
            margin-bottom: 10px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="animation-container">
        <div class="star-animation">
            <div class="label">Star & Planet System</div>
            <canvas id="starCanvas" width="500" height="300"></canvas>
        </div>
        
        <div class="graph-container">
            <div class="label">Brightness vs Time (Light Curve)</div>
            <canvas id="graphCanvas" width="500" height="300"></canvas>
        </div>
    </div>

    <div class="controls">
        <button id="playPause">⏸ Pause</button>
        <button id="reset">🔄 Reset</button>
    </div>

    <div class="info">
        <strong>What you're seeing:</strong> As the planet crosses in front of the star (transit), 
        it blocks some light, causing a dip in brightness on the graph. 
        This repeating pattern is how we detect exoplanets!
    </div>

    <script>
        const starCanvas = document.getElementById('starCanvas');
        const graphCanvas = document.getElementById('graphCanvas');
        const starCtx = starCanvas.getContext('2d');
        const graphCtx = graphCanvas.getContext('2d');
        
        let animationId;
        let isPlaying = true;
        let time = 0;
        const dataPoints = [];
        const maxDataPoints = 200;
        
        const star = {
            x: starCanvas.width / 2,
            y: starCanvas.height / 2,
            radius: 60
        };
        
        const planet = {
            radius: 15,
            orbitRadius: 200,
            speed: 0.015,
            angle: -Math.PI
        };
        
        const graphPadding = 40;
        const graphWidth = graphCanvas.width - 2 * graphPadding;
        const graphHeight = graphCanvas.height - 2 * graphPadding;
        
        function drawStar() {
            const gradient = starCtx.createRadialGradient(star.x, star.y, 0, star.x, star.y, star.radius * 1.5);
            gradient.addColorStop(0, 'rgba(255, 220, 100, 0.8)');
            gradient.addColorStop(0.5, 'rgba(255, 200, 50, 0.4)');
            gradient.addColorStop(1, 'rgba(255, 180, 0, 0)');
            
            starCtx.fillStyle = gradient;
            starCtx.beginPath();
            starCtx.arc(star.x, star.y, star.radius * 1.5, 0, Math.PI * 2);
            starCtx.fill();
            
            starCtx.fillStyle = '#FFD700';
            starCtx.beginPath();
            starCtx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            starCtx.fill();
            
            starCtx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            starCtx.beginPath();
            starCtx.arc(star.x - 15, star.y - 15, 12, 0, Math.PI * 2);
            starCtx.fill();
        }
        
        function drawPlanet(x, y) {
            starCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            starCtx.beginPath();
            starCtx.arc(x + 2, y + 2, planet.radius, 0, Math.PI * 2);
            starCtx.fill();
            
            const gradient = starCtx.createRadialGradient(x - 5, y - 5, 0, x, y, planet.radius);
            gradient.addColorStop(0, '#6B8EC7');
            gradient.addColorStop(1, '#2E5090');
            
            starCtx.fillStyle = gradient;
            starCtx.beginPath();
            starCtx.arc(x, y, planet.radius, 0, Math.PI * 2);
            starCtx.fill();
            
            starCtx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            starCtx.beginPath();
            starCtx.arc(x - 5, y - 5, 4, 0, Math.PI * 2);
            starCtx.fill();
        }
        
        function drawOrbitPath() {
            starCtx.strokeStyle = 'rgba(135, 206, 235, 0.2)';
            starCtx.lineWidth = 1;
            starCtx.setLineDash([5, 5]);
            starCtx.beginPath();
            starCtx.moveTo(star.x - planet.orbitRadius, star.y);
            starCtx.lineTo(star.x + planet.orbitRadius, star.y);
            starCtx.stroke();
            starCtx.setLineDash([]);
        }
        
        function calculateBrightness(planetX, planetY) {
            const distance = Math.sqrt(Math.pow(planetX - star.x, 2) + Math.pow(planetY - star.y, 2));
            
            if (distance < star.radius) {
                const overlap = (star.radius - distance) / star.radius;
                const blockage = (planet.radius * planet.radius) / (star.radius * star.radius);
                return 1 - (blockage * Math.max(0, overlap * 1.5));
            }
            
            return 1.0;
        }
        
        function drawGraph() {
            graphCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            graphCtx.fillRect(0, 0, graphCanvas.width, graphCanvas.height);
            
            graphCtx.strokeStyle = '#87CEEB';
            graphCtx.lineWidth = 2;
            
            graphCtx.beginPath();
            graphCtx.moveTo(graphPadding, graphPadding);
            graphCtx.lineTo(graphPadding, graphCanvas.height - graphPadding);
            graphCtx.stroke();
            
            graphCtx.beginPath();
            graphCtx.moveTo(graphPadding, graphCanvas.height - graphPadding);
            graphCtx.lineTo(graphCanvas.width - graphPadding, graphCanvas.height - graphPadding);
            graphCtx.stroke();
            
            graphCtx.fillStyle = '#E6E6FA';
            graphCtx.font = '12px Roboto Mono';
            graphCtx.textAlign = 'center';
            graphCtx.fillText('Time →', graphCanvas.width / 2, graphCanvas.height - 10);
            
            graphCtx.save();
            graphCtx.translate(15, graphCanvas.height / 2);
            graphCtx.rotate(-Math.PI / 2);
            graphCtx.fillText('Brightness →', 0, 0);
            graphCtx.restore();
            
            graphCtx.strokeStyle = 'rgba(135, 206, 235, 0.3)';
            graphCtx.lineWidth = 1;
            graphCtx.setLineDash([3, 3]);
            graphCtx.beginPath();
            graphCtx.moveTo(graphPadding, graphPadding + 20);
            graphCtx.lineTo(graphCanvas.width - graphPadding, graphPadding + 20);
            graphCtx.stroke();
            graphCtx.setLineDash([]);
            
            if (dataPoints.length > 1) {
                graphCtx.strokeStyle = '#4a90e2';
                graphCtx.lineWidth = 2;
                graphCtx.beginPath();
                
                dataPoints.forEach((point, index) => {
                    const x = graphPadding + (index / maxDataPoints) * graphWidth;
                    const y = graphCanvas.height - graphPadding - (point * graphHeight * 0.8);
                    
                    if (index === 0) {
                        graphCtx.moveTo(x, y);
                    } else {
                        graphCtx.lineTo(x, y);
                    }
                });
                
                graphCtx.stroke();
                
                const lastPoint = dataPoints[dataPoints.length - 1];
                const lastX = graphPadding + ((dataPoints.length - 1) / maxDataPoints) * graphWidth;
                const lastY = graphCanvas.height - graphPadding - (lastPoint * graphHeight * 0.8);
                
                graphCtx.fillStyle = '#FFD700';
                graphCtx.beginPath();
                graphCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
                graphCtx.fill();
            }
        }
        
        function animate() {
            if (!isPlaying) return;
            
            planet.angle += planet.speed;
            if (planet.angle > Math.PI) {
                planet.angle = -Math.PI;
            }
            
            const planetX = star.x + Math.cos(planet.angle) * planet.orbitRadius;
            const planetY = star.y;
            
            starCtx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            starCtx.fillRect(0, 0, starCanvas.width, starCanvas.height);
            
            drawOrbitPath();
            drawStar();
            drawPlanet(planetX, planetY);
            
            const brightness = calculateBrightness(planetX, planetY);
            dataPoints.push(brightness);
            
            if (dataPoints.length > maxDataPoints) {
                dataPoints.shift();
            }
            
            drawGraph();
            
            time++;
            animationId = requestAnimationFrame(animate);
        }
        
        document.getElementById('playPause').addEventListener('click', function() {
            isPlaying = !isPlaying;
            this.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            if (isPlaying) {
                animate();
            }
        });
        
        document.getElementById('reset').addEventListener('click', function() {
            planet.angle = -Math.PI;
            time = 0;
            dataPoints.length = 0;
            starCtx.clearRect(0, 0, starCanvas.width, starCanvas.height);
            graphCtx.clearRect(0, 0, graphCanvas.width, graphCanvas.height);
            drawGraph();
        });
        
        animate();
    </script>
</body>
</html>
""", height=650)
st.markdown("---")
st.header("Real-World Light Curve Examples")
st.image(
    "photo-collage.jpg",
    caption="Comparison of light curves. The top graph shows distinct, periodic dips characteristic of an exoplanet transit. The bottom graph displays stellar variability and noise without a clear transit signal."
)
st.markdown("---")

st.header("About This Application")
st.markdown(
    """
    This is a professional interface for a suite of AI models designed to detect
    exoplanets from astrophysical data. Use the navigation sidebar to access the analysis tools.
    ### Important Parameters to Know:

    ### Period (The Rhythm):
    - *What it is:* How often the light dims.
    - *Why it matters:* A real planet has a perfect, repeating rhythm. If the light dims exactly every 3 days, it's a good sign. If the timing is random, it's not a planet.
    ### Duration (The Crossing Time):
    - *What it is:* How long the light stays dim each time.
    - *Why it matters:* The time it takes for the bug to cross the light bulb tells us how fast it's moving. A dip that's too quick is likely just a glitch.
    ### Depth (The Dimness):
    - *What it is:* How much the light dims.
    - *Why it matters:* This tells you the size of the object. A tiny bug (a planet) will only make the light a tiny bit dimmer. If the light gets very dim, the object was probably another massive light bulb (another star), not a planet.
    ### Signal-to-Noise Ratio / SNR (The Clarity):
    - *What it is:* How clear and obvious the dimming is.
    - *Why it matters:* A sharp, clear dip is a believable signal (High SNR). A fuzzy, faint dip that's hard to see against the background flicker is probably not real (Low SNR).
    ### Stellar Parameters (Info about the Star):
    - *What they are:* Knowing details about the star itself (Is it big? Is it hot?).
    - *Why they matter:* If you know you're looking at a small star, but you see a shadow that suggests a giant planet, something is wrong. Knowing about the star helps the AI check if the story adds up.

    ### Available Tools:
    - *Light Curve Analysis*: Upload a FITS file and use a deep learning (CNN) model to detect transit signals.
    - *Tabular Model Analysis*: Input stellar and transit parameters or CSV file to get a prediction from a powerful XGBoost model.
    - *FITS Viewer*: A utility to inspect the contents of any FITS file.
    - *Admin Tools*: A protected area for managing the AI models.
    """
)

# --- ANIMATED FLOWCHART (HTML ONLY, NO STYLE TAG) ---
st.markdown("---")
st.header("Software Architecture & Data Flow")

flowchart_html = """
<div class="flowchart-container">
    <div id="layer-1" class="flowchart-layer">
        <div class="flowchart-node">☁ Cloud Storage & APIs</div>
        <div class="flowchart-node">🗃 FITS Files & Databases</div>
    </div>
    <div id="arrow-1" class="flowchart-arrow">⬇</div>
    <div id="layer-2" class="flowchart-layer">
        <div class="flowchart-node">🧹 Data Preprocessing</div>
        <div class="flowchart-node ai-core">🧠 AI CORE (CNN & XGBoost Models)</div>
        <div class="flowchart-node">📈 Predictive Analytics</div>
    </div>
    <div id="arrow-2" class="flowchart-arrow">⬇</div>
    <div id="layer-3" class="flowchart-layer">
        <div class="flowchart-node">📉 Light Curve Analysis</div>
        <div class="flowchart-node">📊 Tabular Model Analysis</div>
        <div class="flowchart-node">🔭 FITS Viewer</div>
    </div>
    <div id="arrow-3" class="flowchart-arrow">⬇</div>
    <div id="layer-4" class="flowchart-layer">
        <div class="flowchart-node">🖥 Professional User Interface</div>
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








