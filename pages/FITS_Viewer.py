import streamlit as st
from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import lightkurve as lk
import base64
from styling import add_advanced_loading_animation, load_custom_styling_back


add_advanced_loading_animation()
load_custom_styling_back()
# --- Page Config & Background ---
st.set_page_config(layout="wide")

@st.cache_data
def get_base_64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

def set_png_as_page_bg(png_file):
    bin_str = get_base_64_of_bin_file(png_file)
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
    st.sidebar.warning("Background image 'background.jpg' not found.")


# --- Main App ---
st.title("🔭 FITS File Viewer")
st.write("Upload a FITS file to inspect its headers, data tables, images, and light curve plots.")

uploaded_file = st.file_uploader("Upload or Drag and Drop your FITS file", type=["fits", "fit", "fz"])

if uploaded_file is not None:
    try:
        # Open the FITS file from the uploaded buffer
        hdul = fits.open(uploaded_file)
        st.success(f"Successfully opened **{uploaded_file.name}**")
        
        hdu_options = [f"HDU {i}: {hdul[i].name}" for i in range(len(hdul))]
        # Set default to inspect the first HDU, or all if less than 3
        default_selection = hdu_options if len(hdu_options) <= 3 else hdu_options[0]
        selected_hdu_names = st.multiselect("Select one or more HDUs to inspect:", hdu_options, default=default_selection)

        if selected_hdu_names:
            for hdu_name in selected_hdu_names:
                selected_hdu_index = hdu_options.index(hdu_name)
                hdu = hdul[selected_hdu_index]

                st.markdown("---")
                st.header(f"Inspecting {hdu_name}")

                with st.expander("Show Header Metadata"):
                    header_str = repr(hdu.header)
                    st.code(header_str, language='text')

                st.subheader("Data Content")
                if hdu.data is None:
                    st.warning("This HDU contains no data.")
                
                elif isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)) and hdu.data.ndim >= 2:
                    st.write(f"Displaying Image Data (Shape: {hdu.data.shape})")
                    col1, col2 = st.columns(2)
                    cmap = col1.selectbox("Color Map", plt.colormaps(), key=f"cmap_{selected_hdu_index}")
                    # UPGRADED: More scaling options
                    scale = col2.selectbox("Image Scale", ["Linear", "Log", "Square Root"], key=f"scale_{selected_hdu_index}")
                    
                    image_data = hdu.data.astype(np.float32)
                    # Handle negative values before log/sqrt scaling
                    min_val = np.nanmin(image_data)
                    image_data -= min_val 
                    
                    if scale == 'Log':
                        image_data = np.log1p(image_data)
                    elif scale == 'Square Root':
                        image_data = np.sqrt(image_data)

                    fig, ax = plt.subplots()
                    ax.imshow(image_data, cmap=cmap, origin="lower")
                    ax.set_facecolor('black')
                    ax.set_title(f"HDU {selected_hdu_index} Image Data")
                    st.pyplot(fig)

                elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    data_columns = [col.upper() for col in hdu.data.names]
                    time_cols = ['TIME', 'BARYTIME', 'TTIME', 'JD']
                    flux_cols = ['FLUX', 'SAP_FLUX', 'PDCSAP_FLUX', 'KSPSAP_FLUX']
                    
                    time_col_found = next((col for col in time_cols if col in data_columns), None)
                    flux_col_found = next((col for col in flux_cols if col in data_columns), None)
                    
                    if time_col_found and flux_col_found:
                        st.info(f"Light curve data found! Displaying analysis plots.")
                        
                        full_df = pd.DataFrame.from_records(hdu.data)
                        df = full_df[[time_col_found, flux_col_found]].dropna()
                        
                        # Create a lightkurve object for analysis
                        lc = lk.LightCurve(time=df[time_col_found], flux=df[flux_col_found])
                        lc_flat = lc.flatten()
                        
                        # --- NEW: TABBED PLOT SUITE ---
                        lc_tabs = st.tabs(["Folded Light Curve", "Periodogram", "Full Light Curve", "Raw Data Table"])
                        
                        with lc_tabs[0]:
                            st.subheader("Folded Light Curve")
                            try:
                                periodogram = lc_flat.to_periodogram(method='bls')
                                best_period = periodogram.period_at_max_power
                                folded_lc = lc_flat.fold(period=best_period)
                                fig_folded = px.scatter(x=folded_lc.time.value, y=folded_lc.flux.value, title=f"Light Curve Folded at Best Period ({best_period.value:.4f} days)")
                                st.plotly_chart(fig_folded, use_container_width=True)
                                st.success(f"The strongest periodic signal was found at {best_period.value:.4f} days. This plot shows all transits stacked on top of each other.")
                            except Exception as e:
                                st.warning(f"Could not generate a folded light curve: {e}")
                        
                        with lc_tabs[1]:
                            st.subheader("Periodogram (Signal Search)")
                            try:
                                periodogram = lc_flat.to_periodogram(method='bls')
                                fig_periodogram = px.line(x=periodogram.period.value, y=periodogram.power.value, title="Box-Least-Squares (BLS) Periodogram", labels={'x': 'Period (days)', 'y': 'Power'})
                                st.plotly_chart(fig_periodogram, use_container_width=True)
                                st.info("This plot shows the strength of repeating signals at different periods. The highest peak indicates the most likely transit period.")
                            except Exception as e:
                                st.warning(f"Could not generate a periodogram: {e}")

                        with lc_tabs[2]:
                            st.subheader("Full Light Curve")
                            fig = px.scatter(df, x=time_col_found, y=flux_col_found, title=f"Light Curve from {hdu_name}")
                            fig.update_traces(mode='lines', line=dict(width=1))
                            st.plotly_chart(fig, use_container_width=True)

                        with lc_tabs[3]:
                            st.subheader("Raw Data Table")
                            st.dataframe(full_df)
                    
                    else:
                        st.write(f"Displaying Table Data ({len(hdu.data)} rows)")
                        df = pd.DataFrame.from_records(hdu.data)
                        st.dataframe(df)
                    
                else:
                    st.info("This HDU contains data that is not a standard image or table.")

    except Exception as e:

        st.error(f"An error occurred while reading the FITS file: {e}")


