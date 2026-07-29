import streamlit as st
import pickle
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Chemical Treatment Optimization", layout="wide")
st.title("Chemical Treatment Optimization")
st.markdown("Optimize chemical dosage for well treatment.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("dosage", "dosage_optimizer.pkl"), ("effectiveness", "effectiveness_predictor.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
treatment_type = st.sidebar.selectbox("Treatment Type", ['scale_inhibitor', 'corrosion_inhibitor', 'demulsifier', 'paraffin_inhibitor'])
temperature_c = st.sidebar.slider("Temperature C", 20, 150, 85)
ph = st.sidebar.slider("Ph", 3, 12, 7)
water_hardness = st.sidebar.slider("Water Hardness", 50, 500, 275)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[treatment_type, temperature_c, ph, water_hardness]])
        m = models["dosage"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Dosage", result if isinstance(result, str) else f"{result:.4f}")
        m = models["effectiveness"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Effectiveness", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")
