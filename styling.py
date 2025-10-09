import streamlit as st
import time

def add_advanced_loading_animation():
    """
    A more sophisticated loading animation with a progress bar effect.
    PLUS enhanced visual effects for the entire application.
    """
    loading_html = """
    <style>
        #loading-overlay-advanced {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #000428 0%, #000000 100%);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            animation: fadeOut 0.8s ease-in-out 3s forwards;
        }
        
        /* Orbital Spinner */
        .orbital-spinner {
            position: relative;
            width: 100px;
            height: 100px;
        }
        
        .orbit {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 3px solid transparent;
            border-top-color: #87CEEB;
            border-radius: 50%;
            animation: orbit 1.5s linear infinite;
        }
        
        .orbit:nth-child(2) {
            width: 70%;
            height: 70%;
            top: 15%;
            left: 15%;
            border-top-color: #4a90e2;
            animation-duration: 1s;
            animation-direction: reverse;
        }
        
        .orbit:nth-child(3) {
            width: 40%;
            height: 40%;
            top: 30%;
            left: 30%;
            border-top-color: #E6E6FA;
            animation-duration: 0.7s;
        }
        
        /* Progress Bar */
        .progress-container {
            width: 300px;
            height: 4px;
            background: rgba(135, 206, 235, 0.2);
            border-radius: 2px;
            margin-top: 40px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #87CEEB, #4a90e2, #87CEEB);
            background-size: 200% 100%;
            animation: progress 2.5s ease-in-out forwards, shimmer 1s linear infinite;
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(135, 206, 235, 0.5);
        }
        
        /* Loading Text */
        .loading-text-advanced {
            margin-top: 25px;
            font-family: 'Orbitron', 'Roboto Mono', monospace;
            font-size: 20px;
            color: #E6E6FA;
            text-shadow: 0 0 10px rgba(135, 206, 235, 0.5);
            letter-spacing: 2px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes orbit {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes progress {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        @keyframes fadeOut {
            to {
                opacity: 0;
                visibility: hidden;
            }
        }
        
        /* ========== ENHANCED VISUAL EFFECTS FOR THE APP ========== */
        
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
        
        /* Global Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes glow {
            0%, 100% {
                box-shadow: 0 0 5px rgba(99, 102, 241, 0.5),
                            0 0 10px rgba(99, 102, 241, 0.3),
                            0 0 15px rgba(99, 102, 241, 0.2);
            }
            50% {
                box-shadow: 0 0 10px rgba(99, 102, 241, 0.8),
                            0 0 20px rgba(99, 102, 241, 0.6),
                            0 0 30px rgba(99, 102, 241, 0.4);
            }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
        }
        
        /* Enhanced Title Styling */
        h1 {
            font-family: 'Orbitron', sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 0.8s ease-out, shimmer 3s infinite linear;
            background-size: 200% auto;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            letter-spacing: 2px;
            font-weight: 900 !important;
        }
        
        /* Headers Animation */
        h2, h3 {
            font-family: 'Rajdhani', sans-serif !important;
            animation: fadeInUp 0.6s ease-out;
            color: #e0e7ff !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
        }
        
        /* Enhanced Buttons */
        .stButton > button {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-size: 1rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.5s ease-out;
        }
        
        .stButton > button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover:before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6),
                        0 0 30px rgba(102, 126, 234, 0.4);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(1.02);
        }
        
        /* Primary Button Special Effect */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            animation: glow 2s infinite, fadeInUp 0.5s ease-out;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
            box-shadow: 0 8px 30px rgba(245, 87, 108, 0.6),
                        0 0 40px rgba(240, 147, 251, 0.4);
        }
        
        /* Enhanced Radio Buttons */
        .stRadio > div {
            animation: slideInRight 0.6s ease-out;
            gap: 1rem;
        }
        
        .stRadio > div > label {
            font-family: 'Rajdhani', sans-serif !important;
            background: rgba(99, 102, 241, 0.1);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            font-weight: 600;
        }
        
        .stRadio > div > label:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.3);
        }
        
        /* Enhanced Input Fields */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            font-family: 'Rajdhani', sans-serif !important;
            background: rgba(30, 30, 60, 0.6) !important;
            border: 2px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 10px !important;
            color: #e0e7ff !important;
            transition: all 0.3s ease !important;
            padding: 0.75rem !important;
            font-weight: 500 !important;
        }
        
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: rgba(102, 126, 234, 0.8) !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.4) !important;
            transform: scale(1.02);
        }
        
        /* Enhanced Metrics */
        [data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 2rem !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulse 2s infinite, float 3s ease-in-out infinite;
            font-weight: 900 !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-family: 'Rajdhani', sans-serif !important;
            color: #a5b4fc !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }
        
        /* Enhanced Expanders */
        .streamlit-expanderHeader {
            font-family: 'Rajdhani', sans-serif !important;
            background: rgba(99, 102, 241, 0.15) !important;
            border-radius: 10px !important;
            border: 2px solid rgba(99, 102, 241, 0.3) !important;
            transition: all 0.3s ease !important;
            font-weight: 700 !important;
            padding: 1rem !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(99, 102, 241, 0.25) !important;
            border-color: rgba(102, 126, 234, 0.6) !important;
            box-shadow: 0 5px 20px rgba(99, 102, 241, 0.3) !important;
            transform: translateX(5px);
        }
        
        /* Enhanced Data Frames */
        [data-testid="stDataFrame"] {
            animation: fadeInUp 0.6s ease-out;
            border-radius: 15px !important;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        /* Enhanced File Uploader */
        [data-testid="stFileUploader"] {
            animation: fadeInUp 0.7s ease-out;
        }
        
        [data-testid="stFileUploader"] > div {
            background: rgba(99, 102, 241, 0.1) !important;
            border: 2px dashed rgba(99, 102, 241, 0.4) !important;
            border-radius: 15px !important;
            transition: all 0.3s ease !important;
            padding: 2rem !important;
        }
        
        [data-testid="stFileUploader"] > div:hover {
            background: rgba(99, 102, 241, 0.2) !important;
            border-color: rgba(102, 126, 234, 0.7) !important;
            box-shadow: 0 5px 25px rgba(99, 102, 241, 0.3) !important;
            transform: scale(1.02);
        }
        
        /* Enhanced Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(15, 15, 35, 0.95) !important;
            border-right: 2px solid rgba(99, 102, 241, 0.3) !important;
            animation: slideInRight 0.5s ease-out;
        }
        
        [data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, transparent 100%);
        }
        
        /* Enhanced Success/Error/Warning Messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            animation: fadeInUp 0.5s ease-out !important;
            border-radius: 12px !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Enhanced Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
            background-size: 200% auto !important;
            animation: shimmer 2s infinite linear !important;
            border-radius: 10px !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        /* Enhanced Spinner */
        .stSpinner > div {
            border-color: rgba(102, 126, 234, 0.3) !important;
            border-top-color: #667eea !important;
            animation: rotate 1s linear infinite !important;
        }
        
        /* Enhanced Form */
        .stForm {
            animation: fadeInUp 0.6s ease-out;
            background: rgba(99, 102, 241, 0.05) !important;
            border: 2px solid rgba(99, 102, 241, 0.2) !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Enhanced Columns */
        [data-testid="column"] {
            animation: fadeInUp 0.5s ease-out;
        }
        
        /* Markdown Horizontal Rule */
        hr {
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent) !important;
            margin: 2rem 0 !important;
            animation: shimmer 3s infinite linear !important;
            background-size: 200% auto !important;
        }
        
        /* Enhanced Selectbox */
        .stSelectbox > div > div {
            animation: fadeInUp 0.5s ease-out;
        }
        
        /* Plotly Charts Animation */
        .js-plotly-plot {
            animation: fadeInUp 0.8s ease-out;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        }
        
        /* Password Input Special Effect */
        input[type="password"] {
            background: rgba(99, 102, 241, 0.1) !important;
            border: 2px solid rgba(240, 147, 251, 0.4) !important;
            animation: glow 2s infinite;
        }
        
        /* Balloons Enhancement (when triggered) */
        [data-testid="stBalloons"] {
            filter: brightness(1.5) saturate(1.5);
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 15, 35, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            border: 2px solid rgba(15, 15, 35, 0.5);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        }
        
        /* Multiselect Enhancement */
        .stMultiSelect > div > div {
            background: rgba(30, 30, 60, 0.6) !important;
            border: 2px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        .stMultiSelect > div > div:hover {
            border-color: rgba(102, 126, 234, 0.6) !important;
            box-shadow: 0 5px 20px rgba(99, 102, 241, 0.3) !important;
        }
        
        /* Slider Enhancement */
        .stSlider > div > div {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .stSlider [role="slider"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Container Animation */
        .block-container {
            animation: fadeInUp 0.5s ease-out;
        }
        
        /* Footer fade effect */
        footer {
            animation: fadeInUp 1s ease-out;
        }
    </style>
    
    <div id="loading-overlay-advanced">
        <div class="orbital-spinner">
            <div class="orbit"></div>
            <div class="orbit"></div>
            <div class="orbit"></div>
        </div>
        <div class="progress-container">
            <div class="progress-bar"></div>
        </div>
        <div class="loading-text-advanced">
            INITIALIZING SYSTEM
        </div>
    </div>
    """
    
    st.markdown(loading_html, unsafe_allow_html=True)
