import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Chemical Treatment", layout="wide")
st.title("Chemical Treatment")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'dosage': joblib.load(p / 'dosage_optimizer.pkl'), 'effectiveness': joblib.load(p / 'effectiveness_predictor.pkl')}

tab1, tab2, tab3 = st.tabs(['Predict', 'Charts', 'Info'])

with tab1:
    st.subheader('Inputs')
    c = st.columns(2)
    treatment = c[0].selectbox('Treatment', ['scale','corrosion','demulsifier','paraffin'])
    temp = c[1].slider('Temp', 20, 150, 85)
    ph = c[0].slider('Ph', 3, 12, 7)
    hardness = c[1].slider('Hardness', 50, 500, 275)
    if st.button('Run', type='primary'):
        x = np.array([[treatment, temp, ph, hardness]])
        res = {}
        m = models['dosage']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['dosage'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['dosage'] = float(m.predict(x)[0])
        m = models['effectiveness']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['effectiveness'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['effectiveness'] = float(m.predict(x)[0])
        st.divider()
        rc = st.columns(len(res))
        for i, (k, v) in enumerate(res.items()):
            rc[i].metric(k.replace('_',' ').title(), str(v) if isinstance(v,str) else f'{v:.2f}')

with tab2:
    st.info('Charts update after prediction')

with tab3:
    st.markdown('Bayesian optimization of chemical dosage')
    st.caption('Built with scikit-learn + Streamlit')