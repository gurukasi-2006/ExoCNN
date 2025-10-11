import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from styling import add_advanced_loading_animation, load_custom_styling_back

# --- Advanced loading + custom styling (from your existing utilities) ---
add_advanced_loading_animation()
load_custom_styling_back()

# --- Page Configuration ---
st.set_page_config(
    page_title="Model Metrics Visualizer",
    page_icon="📈",
    layout="wide"
)

# --- NASA Theme Background ---
def set_nasa_bg(png_file='space_bg.jpg'):
    try:
        with open(png_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                              url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            color: #E0E0E0;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        st.warning("Background not found, default background applied.")

set_nasa_bg()

# --- Title ---
st.title("🚀 Training Metrics Visualizer")
st.markdown("---")

# --- Load metrics CSV ---
try:
    metrics_df = pd.read_csv("metrics_log.csv")
    st.success("Metrics loaded successfully ✅")
except FileNotFoundError:
    st.error("metrics_log.csv not found. Please run training first.")
    st.stop()

# --- Interactive Metric Selector ---
metric_option = st.selectbox("Choose metric to visualize:", ["Train Loss", "Val Loss", "Train Acc", "Val Acc", "All"])

# --- Plotting ---
plt.style.use('dark_background')
sns.set_palette("coolwarm")
fig, ax = plt.subplots(figsize=(10,5))

if metric_option == "All":
    ax.plot(metrics_df['epoch'], metrics_df['train_loss'], label="Train Loss", color="#FF4500", linestyle='--')
    ax.plot(metrics_df['epoch'], metrics_df['val_loss'], label="Val Loss", color="#1E90FF", linestyle='--')
    ax.plot(metrics_df['epoch'], metrics_df['train_acc'], label="Train Acc", color="#ADFF2F")
    ax.plot(metrics_df['epoch'], metrics_df['val_acc'], label="Val Acc", color="#FFD700")
else:
    col_map = {
        "Train Loss": "#FF4500",
        "Val Loss": "#1E90FF",
        "Train Acc": "#ADFF2F",
        "Val Acc": "#FFD700"
    }
    ax.plot(metrics_df['epoch'], metrics_df[metric_option.lower().replace(" ", "_")], label=metric_option, color=col_map[metric_option])

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Value", fontsize=12, fontweight='bold')
ax.set_title("Training Metrics over Epochs", fontsize=16, fontweight='bold', color="#87CEEB")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- Summary Cards (NASA Style) ---
st.markdown("---")
st.subheader("📊 Quick Stats")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Max Train Acc", f"{metrics_df['train_acc'].max()*100:.2f}%")
col2.metric("Max Val Acc", f"{metrics_df['val_acc'].max()*100:.2f}%")
col3.metric("Min Train Loss", f"{metrics_df['train_loss'].min():.4f}")
col4.metric("Min Val Loss", f"{metrics_df['val_loss'].min():.4f}")

# --- Optional: Preview saved plot ---
st.markdown("---")
st.subheader("Preview Saved Training Graph")
try:
    from PIL import Image
    img = Image.open("training_metrics_preview.png")
    st.image(img, use_container_width=True)
except FileNotFoundError:
    st.warning("training_metrics_preview.png not found.")

