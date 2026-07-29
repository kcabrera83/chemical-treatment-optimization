import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Chemical Treatment Optimization", layout="wide")
st.title("Chemical Treatment Optimization")
st.markdown("Optimize chemical dosage for well treatment.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'dosage': joblib.load(d / 'dosage_optimizer.pkl'), 'effectiveness': joblib.load(d / 'effectiveness_predictor.pkl')}

st.sidebar.header("Input Parameters")
treatment_type = st.sidebar.selectbox('Treatment Type', ['scale_inhibitor','corrosion_inhibitor','demulsifier','paraffin_inhibitor'])
temperature_c = st.sidebar.slider('Temperature C', 20, 150, 85)
ph = st.sidebar.slider('Ph', 3, 12, 7)
water_hardness = st.sidebar.slider('Water Hardness', 50, 500, 275)

if st.sidebar.button("Run"):
    try:
        x = np.array([[treatment_type, temperature_c, ph, water_hardness]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))