import streamlit as st
import time
import base64
import os

#Loading animation function
def add_advanced_loading_animation():
    """
    A sophisticated loading animation that stays until the page fully loads.
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
            transition: opacity 0.8s ease-in-out, visibility 0.8s ease-in-out;
        }
        
        #loading-overlay-advanced.loaded {
            opacity: 0;
            visibility: hidden;
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
            animation: shimmer 1s linear infinite;
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(135, 206, 235, 0.5);
            width: 0%;
            transition: width 0.3s ease-out;
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
        
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
    </style>
    
    <div id="loading-overlay-advanced">
        <div class="orbital-spinner">
            <div class="orbit"></div>
            <div class="orbit"></div>
            <div class="orbit"></div>
        </div>
        <div class="progress-container">
            <div class="progress-bar" id="progress-bar"></div>
        </div>
        <div class="loading-text-advanced">
            INITIALIZING SYSTEM
        </div>
    </div>
    
    <script>
        // Check if this is the first load using sessionStorage flag
        const hasLoadedBefore = sessionStorage.getItem('hasLoaded');
        
        if (hasLoadedBefore) {
            // Already loaded once, hide immediately
            document.getElementById('loading-overlay-advanced').style.display = 'none';
        } else {
            // First load - show animation
            let progress = 0;
            const progressBar = document.getElementById('progress-bar');
            const overlay = document.getElementById('loading-overlay-advanced');
            
            // Update progress bar
            const progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.random() * 10;
                    progress = Math.min(progress, 90);
                    progressBar.style.width = progress + '%';
                }
            }, 100);
            
            // Hide loading screen when page is fully loaded
            window.addEventListener('load', () => {
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                
                setTimeout(() => {
                    overlay.classList.add('loaded');
                    sessionStorage.setItem('hasLoaded', 'true');
                }, 300);
            });
            
            // Fallback: hide after 3 seconds
            setTimeout(() => {
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                overlay.classList.add('loaded');
                sessionStorage.setItem('hasLoaded', 'true');
            }, 3000);
        }
    </script>
    """
    st.markdown(loading_html, unsafe_allow_html=True)


def get_image_as_base64_back(path):
    """Gets the base64 string of an image file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def load_custom_styling_back():
    """Injects all custom CSS for the entire page in one block."""
    # Encode the local sidebar background to base64
    sidebar_bg_path = "exo_assets/sidebar_background.jpg"
    sidebar_bg_base64 = get_image_as_base64_back(sidebar_bg_path)

    
    if sidebar_bg_base64:
        sidebar_style = f'''
            [data-testid="stSidebar"] > div:first-child {{
                background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("data:image/jpeg;base64,{sidebar_bg_base64}");
                background-position: center; 
                background-repeat: no-repeat;
                background-size: cover;
            }}
        '''
    else:
        sidebar_style = "" 

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap');

        /* --- Sidebar Styling --- */
        {sidebar_style}
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
        
        /* --- (Add any other general styles here if needed) --- */

    </style>
    """, unsafe_allow_html=True)




