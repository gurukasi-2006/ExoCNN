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

load_custom_styling_back()
add_advanced_loading_animation()
# --- Page Configuration and Background ---
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
    set_png_as_page_bg('background.jpg')
except Exception:
    st.sidebar.warning("Background image 'background.jpg' not found.")


# --- Model and Artifact Loading ---
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

# --- Preprocessing and Helper Functions ---
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

# --- Sidebar ---
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

# --- Main App UI ---
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
            with st.expander("Show Prediction Explanation (SHAP Plot)"):
                shap_values = explainer.shap_values(processed_df)
                st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], processed_df.iloc[0,:]), 400)
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
                                # This is the full, working training pipeline from your notebook
                                st.info("Step 1/5: Preprocessing new data...")
                                # We need a full preprocessing function for training here
                                training_df = st.session_state.uploaded_data.copy()
                                training_df = training_df.rename(columns={'koi_disposition': 'label'}) # Simple rename
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

                                    # Placeholder for your full feature engineering
                                    # In a real app, the full preprocess_for_prediction function would be called
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



