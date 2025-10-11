import streamlit as st
import time

def add_advanced_loading_animation():
    """
    A more sophisticated loading animation with a progress bar effect.

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



































































































































































































































































































































































































