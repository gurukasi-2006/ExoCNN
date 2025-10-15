import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import base64
import shap
import plotly.express as px
import os
import shutil
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from styling import add_advanced_loading_animation, load_custom_styling_back
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import matplotlib.pyplot as plt

add_advanced_loading_animation()
load_custom_styling_back()
#Page Configuration and Backgroun ---
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
            background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_png_as_page_bg('exo_assets/background.jpg')
except Exception:
    st.sidebar.warning("Background image 'background.jpg' not found.")


# Model and Artifact Loading
@st.cache_resource
def load_artifacts():
    try:
        artifacts_path = "Tabular_Model_Artifacts"
        model = joblib.load(os.path.join(artifacts_path, "exoplanet_model_koi_final_original.pkl"))
        encoder = joblib.load(os.path.join(artifacts_path, "label_encoder_final_original.pkl"))
        scaler = joblib.load(os.path.join(artifacts_path, "scaler_final_original.pkl"))
        explainer = shap.TreeExplainer(model)
        return model, encoder, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"CRITICAL ERROR: A required model file was not found: {e.filename}. Please place all .pkl files in the 'Tabular_Model_Artifacts' folder.")
        return None, None, None, None

model, le, scaler, explainer = load_artifacts()

#Preprocessing and Helper Functions
def preprocess_for_prediction(df, dataset_type):
    df = df.copy()
    column_map = {
        'koi': {'koi_period': 'period', 'koi_time0bk': 'epoch', 'koi_duration': 'duration', 'koi_depth': 'depth', 'koi_prad': 'prad', 'koi_srad': 'srad', 'koi_steff': 'teff', 'koi_slogg': 'logg', 'koi_smass': 'smass', 'koi_model_snr': 'snr', 'koi_disposition': 'label'},
        'k2': {'pl_orbper': 'period', 'pl_tranmid': 'epoch', 'pl_trandur': 'duration', 'pl_trandep': 'depth', 'pl_rade': 'prad', 'st_rad': 'srad', 'st_teff': 'teff', 'st_logg': 'logg', 'st_mass': 'smass', 'pl_snr': 'snr', 'disposition': 'label'},
        'tess': {'pl_orbper': 'period', 'pl_tranmid': 'epoch', 'pl_trandurh': 'duration', 'pl_trandep': 'depth', 'pl_rade': 'prad', 'st_rad': 'srad', 'st_teff': 'teff', 'st_logg': 'logg', 'st_mass': 'smass', 'toi_snr': 'snr', 'tfopwg_disp': 'label'}
    }
    if dataset_type.lower() in column_map:
        df = df.rename(columns=column_map[dataset_type.lower()])
    
    core_features = ['period', 'epoch', 'duration', 'depth', 'prad', 'srad', 'teff', 'logg', 'smass', 'snr']
    for col in core_features:
        if col not in df.columns: df[col] = np.nan
            
    for col in core_features:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val): df[col] = df[col].fillna(0)
            else: df[col] = df[col].fillna(median_val)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df['ror'] = df['prad'] / df['srad'].replace(0, np.nan)
    df['depth_per_duration'] = df['depth'] / df['duration'].replace(0, np.nan)
    df['log_period'] = np.log(df['period'].replace(0, np.nan))
    df['log_depth'] = np.log(df['depth'].replace(0, np.nan))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    expected_features = scaler.get_feature_names_out()
    
    for col in expected_features:
        if col not in df.columns: df[col] = 0
            
    df_features = df[expected_features]
    scaled_data = scaler.transform(df_features)
    return pd.DataFrame(scaled_data, columns=expected_features)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

def generate_pdf_report(input_params, prediction_result, confidence, shap_values, processed_df):
    """Generate a PDF report for single target prediction"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
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
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Exoplanet Classification Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Prediction Result Section
    elements.append(Paragraph("Prediction Result", heading_style))
    verdict_color = colors.green if prediction_result == "Confirmed Planet" else colors.red
    result_data = [
        ['Verdict:', prediction_result],
        ['Confidence Score:', f'{confidence:.2%}'],
        ['Model Accuracy:', f'{st.session_state.champion_accuracy:.4f}']
    ]
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (1, 0), (1, 0), verdict_color),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 20))
    
    # Input Parameters Section
    elements.append(Paragraph("Input Parameters", heading_style))
    
    # Transit & Orbital Properties
    elements.append(Paragraph("<b>Transit & Orbital Properties</b>", styles['Heading3']))
    transit_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Orbital Period', f"{input_params['period']:.4f}", 'days'],
        ['Transit Depth', f"{input_params['depth'] * 1_000_000:.1f}", 'ppm'],
        ['Transit Duration', f"{input_params['duration']:.2f}", 'hours'],
        ['Planet Radius', f"{input_params['prad']:.2f}", 'Earth Radii'],
        ['Epoch', f"{input_params['epoch']:.4f}", 'BJD'],
        ['Signal-to-Noise Ratio', f"{input_params['snr']:.2f}", '']
    ]
    transit_table = Table(transit_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    transit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
    ]))
    elements.append(transit_table)
    elements.append(Spacer(1, 15))
    
    # Stellar Properties
    elements.append(Paragraph("<b>Stellar Properties</b>", styles['Heading3']))
    stellar_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Stellar Radius', f"{input_params['srad']:.2f}", 'Solar Radii'],
        ['Effective Temperature', f"{input_params['teff']:.0f}", 'K'],
        ['Stellar Mass', f"{input_params['smass']:.2f}", 'Solar Mass'],
        ['Surface Gravity', f"{input_params['logg']:.2f}", 'log10(g)']
    ]
    stellar_table = Table(stellar_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    stellar_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
    ]))
    elements.append(stellar_table)
    elements.append(Spacer(1, 20))
    
    # SHAP Plot Section
    elements.append(PageBreak())
    elements.append(Paragraph("Feature Importance Analysis (SHAP)", heading_style))
    elements.append(Paragraph("The SHAP plot below shows how each feature contributed to the prediction:", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Generate SHAP waterfall plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value,
                                             data=processed_df.iloc[0],
                                             feature_names=processed_df.columns.tolist()),
                           max_display=10, show=False)
        
        # Save plot to buffer
        img_buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Add image to PDF
        img = Image(img_buffer, width=6*inch, height=4*inch)
        elements.append(img)
    except Exception as e:
        elements.append(Paragraph(f"<i>SHAP plot generation failed: {str(e)}</i>", styles['Italic']))
    
    elements.append(Spacer(1, 20))
    
    # Footer
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("This report was generated by the ExoCNN-Tabular Analysis", footer_style))
    elements.append(Paragraph("Powered by XGBoost and SHAP", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Sidebar
st.sidebar.header("Model Performance")
if 'champion_accuracy' not in st.session_state:
    try:
        with open("Tabular_Model_Artifacts/accuracy_log.txt", "r") as f:
            st.session_state.champion_accuracy = float(f.read())
    except:
        st.session_state.champion_accuracy = 91.24
with st.sidebar.expander("Show Stats", expanded=True):
    st.write("**Model:** XGBoost Classifier")
    st.metric("Current Model Accuracy", f"{st.session_state.champion_accuracy:.4f}")

# Main App UI
st.title("🌌 Exoplanet Classification System")
if model is None: st.stop()

page = st.radio(
    "Choose an analysis tool:",
    ["Batch Analysis (CSV Upload)", "Single Target Prediction", "⚙️ Admin & Model Management"],
    horizontal=True, label_visibility="collapsed"
)
st.markdown("---")

if page == "Batch Analysis (CSV Upload)":
    st.header("Analyze a Batch of Targets from a CSV File")
    dataset_type = st.selectbox("1. Select the data source type for your CSV:", ["KOI", "K2", "TESS"])
    st.info("💡 Tip: You can also drag and drop your file directly onto the widget below.")
    uploaded_csv = st.file_uploader("2. Upload your CSV file", type=["csv"], key="batch_uploader")
    if uploaded_csv is not None:
        try:
            user_df_raw = pd.read_csv(uploaded_csv, comment='#', engine='python')
            with st.expander("🔎 Explore Uploaded Data"):
                st.dataframe(user_df_raw.describe())
                if len(user_df_raw.columns) > 1:
                    exp_col1, exp_col2 = st.columns(2)
                    x_axis = exp_col1.selectbox("Choose column for X-axis", user_df_raw.columns, index=0)
                    y_axis = exp_col2.selectbox("Choose column for Y-axis", user_df_raw.columns, index=1)
                    if x_axis is not None and y_axis is not None:
                        fig = px.scatter(user_df_raw, x=x_axis, y=y_axis, title=f'Scatter Plot of {y_axis} vs {x_axis}')
                        st.plotly_chart(fig, use_container_width=True)
            label_col_present = any(col in user_df_raw.columns for col in ['koi_disposition', 'label', 'disposition', 'tfopwg_disp'])
            if label_col_present:
                st.success("✅ Labeled data detected. You can now go to the 'Admin & Model Management' tab to use this file for re-training.")
                st.session_state.uploaded_data = user_df_raw
            else:
                st.info("ℹ️ Unlabeled data detected. Ready for batch prediction.")
                if st.button("Process Full File and Predict", type="primary"):
                    with st.spinner("Preprocessing data and predicting for all rows... This may take a moment."):
                        processed_batch = preprocess_for_prediction(user_df_raw, dataset_type)
                        predictions_encoded = model.predict(processed_batch)
                        predictions_labels = le.inverse_transform(predictions_encoded)
                        user_df_raw['model_prediction'] = ["Confirmed Planet" if x == 1.0 else "False Positive" for x in predictions_labels]
                    st.subheader("✅ Analysis Complete!")
                    st.balloons()
                    prediction_summary = user_df_raw['model_prediction'].value_counts()
                    planets_found = prediction_summary.get("Confirmed Planet", 0)
                    fps_found = prediction_summary.get("False Positive", 0)
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("Total Rows Processed", value=len(user_df_raw))
                    stat_cols[1].metric("Planet Candidates Found", value=planets_found)
                    stat_cols[2].metric("False Positives Found", value=fps_found)
                    st.subheader("Full Results with Predictions")
                    st.dataframe(user_df_raw)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

elif page == "Single Target Prediction":
    st.header("Predict for a Single Target")
    with st.form("single_prediction_form"):
        st.subheader("Transit & Orbital Properties")
        col1, col2, col3 = st.columns(3)
        with col1:
            period = st.number_input("Orbital Period (days)", min_value=0.1, value=9.488, step=0.1, format="%.4f")
            depth = st.number_input("Transit Depth (ppm)", min_value=0.1, value=616.0)
        with col2:
            duration = st.number_input("Transit Duration (hours)", min_value=0.1, value=2.96)
            prad = st.number_input("Planet Radius (Earth Radii)", min_value=0.1, value=2.26)
        with col3:
            epoch = st.number_input("Epoch (BJD)", min_value=0.0, value=2459551.0, format="%.4f")
            snr = st.number_input("Signal-to-Noise Ratio (SNR)", min_value=1.0, value=35.80, step=0.1)
        st.subheader("Stellar Properties")
        col4, col5, col6 = st.columns(3)
        with col4:
            srad = st.number_input("Stellar Radius (Solar Radii)", min_value=0.1, value=0.93, step=0.01)
        with col5:
            teff = st.number_input("Stellar Effective Temp (K)", min_value=2000, value=5455)
            smass = st.number_input("Stellar Mass (Solar Mass)", min_value=0.1, value=0.92, step=0.01)
        with col6:
            logg = st.number_input("Stellar Surface Gravity (log10(g))", min_value=1.0, value=4.44, step=0.01)
        submitted = st.form_submit_button("🔮 Run Prediction")

    if submitted:
        raw_data = {
            'period': period, 'epoch': epoch, 'duration': duration, 'depth': depth / 1_000_000,
            'prad': prad, 'srad': srad, 'teff': teff, 'logg': logg, 'smass': smass, 'snr': snr
        }
        input_df = pd.DataFrame([raw_data])
        try:
            processed_df = preprocess_for_prediction(input_df, 'koi')
            prediction_encoded = model.predict(processed_df)[0]
            prediction_proba = model.predict_proba(processed_df)[0]
            prediction_label = le.inverse_transform([prediction_encoded])[0]
            st.subheader("🎯 Prediction Result")
            verdict = "Confirmed Planet" if prediction_label == 'CONFIRMED' else "False Positive"
            confidence = prediction_proba.max()
            if verdict == "Confirmed Planet": st.success(f"**Verdict:** ✅ {verdict}")
            else: st.error(f"**Verdict:** ❌ {verdict}")
            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
            
            # Store prediction results in session state for PDF generation
            st.session_state.last_prediction = {
                'input_params': raw_data,
                'verdict': verdict,
                'confidence': confidence,
                'processed_df': processed_df
            }
            
            with st.expander("Show Prediction Explanation (SHAP Plot)"):
                shap_values = explainer.shap_values(processed_df)
                st.session_state.last_prediction['shap_values'] = shap_values
                st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], processed_df.iloc[0,:]), 400)
            
            # PDF Download Button
            st.markdown("---")
            st.subheader("📄 Download Report")
            
            # Generate PDF automatically after prediction
            try:
                shap_values = st.session_state.last_prediction.get('shap_values')
                if shap_values is None:
                    shap_values = explainer.shap_values(processed_df)
                    st.session_state.last_prediction['shap_values'] = shap_values
                
                with st.spinner("Preparing PDF report..."):
                    pdf_buffer = generate_pdf_report(
                        raw_data,
                        verdict,
                        confidence,
                        shap_values,
                        processed_df
                    )
                    
                    pdf_data = pdf_buffer.getvalue()
                
                # Download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"exoplanet_prediction_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=False
                )
                st.success("✅ PDF report is ready to download!")
                
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                    
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "⚙️ Admin & Model Management":
    st.header("⚙️ Admin & Model Management")
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.warning("⚠️ This section is restricted. Please enter the password to unlock.The Password is : 2030")
        password = st.text_input("Admin Password", type="password", key="admin_password")
        if st.button("🔓 Unlock"):
            if password == "2030":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ The password you entered is incorrect.")
    
    else:
        st.success("✅ Admin Mode Unlocked.")
        
        with st.expander("🔬 Re-train Model with New Labeled Data"):
            st.write("This tool uses the labeled CSV file uploaded in the 'Batch Analysis' tab to train and evaluate a new model.")
            if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
                st.info(f"Ready to train on the uploaded file ({len(st.session_state.uploaded_data)} rows).")
                st.subheader("Hyperparameter Search Space")
                hp_cols = st.columns(3)
                with hp_cols[0]:
                    n_estimators_list = st.multiselect("n_estimators", [100, 200, 300, 400, 500, 700], default=[200, 400])
                with hp_cols[1]:
                    max_depth_range = st.slider("max_depth range", 3, 10, (4, 7))
                with hp_cols[2]:
                    n_iter_search = st.number_input("Number of Search Iterations (n_iter)", min_value=1, max_value=20, value=5)

                with st.expander("⚠️ Are you sure? Re-training can be slow and will overwrite the model if accuracy improves."):
                    if st.button("Confirm and Start Re-training"):
                        with st.spinner("Running full training pipeline... This may take several minutes."):
                            try:
                                
                                st.info("Step 1/5: Preprocessing new data...")
                                
                                training_df = st.session_state.uploaded_data.copy()
                                training_df = training_df.rename(columns={'koi_disposition': 'label'}) 
                                training_df['label'] = training_df['label'].replace({'CONFIRMED': 1.0, 'FALSE POSITIVE': -1.0})
                                training_df = training_df[training_df['label'].isin([1.0, -1.0])]

                                st.info(f"Step 2/5: Balancing data...")
                                confirmed = training_df[training_df['label'] == 1.0]
                                fp = training_df[training_df['label'] == -1.0]
                                n_samples = min(len(confirmed), len(fp))
                                if n_samples < 10:
                                    st.error("Not enough samples of both classes to train.")
                                else:
                                    confirmed_resampled = resample(confirmed, n_samples=n_samples, random_state=42)
                                    fp_resampled = resample(fp, n_samples=n_samples, random_state=42)
                                    balanced_df = pd.concat([confirmed_resampled, fp_resampled])

                                    
                                    X = balanced_df.drop('label', axis=1)
                                    y = balanced_df['label']
                                    
                                    st.info("Step 3/5: Preparing data for training...")
                                    le_new = LabelEncoder().fit(y)
                                    y_enc = le_new.transform(y)
                                    scaler_new = StandardScaler().fit(X)
                                    X_scaled = scaler_new.transform(X)
                                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

                                    param_dist = {
                                        "n_estimators": n_estimators_list,
                                        "max_depth": list(range(max_depth_range[0], max_depth_range[1] + 1)),
                                    }
                                    st.info(f"Step 4/5: Starting RandomizedSearchCV with {n_iter_search} iterations...")
                                    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                                    random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=n_iter_search, cv=3, random_state=42)
                                    random_search.fit(X_train, y_train)
                                    new_model = random_search.best_estimator_
                                    new_accuracy = new_model.score(X_test, y_test)
                                    
                                    st.info("Step 5/5: Evaluating and saving new model...")
                                    if new_accuracy > st.session_state.champion_accuracy:
                                        st.session_state.champion_accuracy = new_accuracy
                                        artifacts_path = "Tabular_Model_Artifacts"
                                        with open(os.path.join(artifacts_path, "accuracy_log.txt"), "w") as f: f.write(str(new_accuracy))
                                        joblib.dump(new_model, os.path.join(artifacts_path, "exoplanet_model_koi_final.pkl"))
                                        joblib.dump(le_new, os.path.join(artifacts_path, "label_encoder_final.pkl"))
                                        joblib.dump(scaler_new, os.path.join(artifacts_path, "scaler_final.pkl"))
                                        st.success(f"Training complete! New accuracy ({new_accuracy:.4f}) is higher. Model files updated.")
                                        st.balloons()
                                        st.cache_resource.clear()
                                    else:
                                        st.warning(f"Training complete, but new accuracy ({new_accuracy:.4f}) did not beat current model ({st.session_state.champion_accuracy:.4f}). Model not updated.")
                            except Exception as e:
                                st.error(f"An error occurred during training: {e}")
            else:
                st.info("Please upload a labeled CSV in the 'Batch Analysis' tab first.")

        with st.expander("⬆️ Upload a New Model File"):
            st.write("Manually replace the current model files. All three files (model, scaler, encoder) are required.")
            artifacts_path = "Tabular_Model_Artifacts"
            new_model_file = st.file_uploader("Upload model .pkl", type=["pkl"], key="up_model")
            new_scaler_file = st.file_uploader("Upload scaler .pkl", type=["pkl"], key="up_scaler")
            new_encoder_file = st.file_uploader("Upload encoder .pkl", type=["pkl"], key="up_encoder")

            if new_model_file and new_scaler_file and new_encoder_file:
                 with st.expander("⚠️ Are you sure you want to overwrite the current model files?"):
                    if st.button("Confirm and Replace All Files"):
                        with open(os.path.join(artifacts_path, "exoplanet_model_koi_final.pkl"), "wb") as f: f.write(new_model_file.getbuffer())
                        with open(os.path.join(artifacts_path, "scaler_final.pkl"), "wb") as f: f.write(new_scaler_file.getbuffer())
                        with open(os.path.join(artifacts_path, "label_encoder_final.pkl"), "wb") as f: f.write(new_encoder_file.getbuffer())
                        st.success("All model files updated. Clearing cache and refreshing app...")
                        st.cache_resource.clear()
                        time.sleep(2); st.rerun()

        with st.expander("🔄 Reset to Original Model"):
            st.warning("This will discard any re-trained progress and restore the original model files.")
            with st.expander("⚠️ This action cannot be undone. Are you absolutely sure?"):
                if st.button("Confirm and Reset Model"):
                    try:
                        artifacts_path = "Tabular_Model_Artifacts"
                        shutil.copyfile(os.path.join(artifacts_path, "exoplanet_model_koi_final_original.pkl"), os.path.join(artifacts_path, "exoplanet_model_koi_final.pkl"))
                        shutil.copyfile(os.path.join(artifacts_path, "scaler_final_original.pkl"), os.path.join(artifacts_path, "scaler_final.pkl"))
                        shutil.copyfile(os.path.join(artifacts_path, "label_encoder_final_original.pkl"), os.path.join(artifacts_path, "label_encoder_final.pkl"))
                        st.success("Model has been reset. Clearing cache and refreshing...")
                        st.cache_resource.clear()
                        time.sleep(2); st.rerun()
                    except FileNotFoundError:
                        st.error("Original backup files (e.g., 'exoplanet_model_koi_original.pkl') not found.")
        
        if st.button("🔒 Lock Admin Mode"):
            st.session_state.authenticated = False

            st.rerun()
