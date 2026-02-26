import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# --- Load and Train Model ---
@st.cache_resource
def train_model():
    df = pd.read_csv('dataset/heart.csv')
    dataset = pd.get_dummies(df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

    scaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

    X = dataset.drop(['target'], axis=1)
    y = dataset['target']

    model = KNeighborsClassifier(n_neighbors=12)
    model.fit(X, y)

    return model, scaler, X.columns.tolist()

model, scaler, feature_cols = train_model()

# --- UI ---
st.title("Heart Disease Risk Predictor")
st.markdown("Enter the patient's clinical details below to predict heart disease risk.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=[
        "0 — Typical Angina",
        "1 — Atypical Angina",
        "2 — Non-Anginal Pain",
        "3 — Asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])
    restecg = st.selectbox("Resting ECG", options=[
        "0 — Normal",
        "1 — ST-T Wave Abnormality",
        "2 — Left Ventricular Hypertrophy"
    ])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=["No (0)", "Yes (1)"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
        "0 — Upsloping",
        "1 — Flat",
        "2 — Downsloping"
    ])
    ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[
        "1 — Normal",
        "2 — Fixed Defect",
        "3 — Reversible Defect"
    ])

st.markdown("---")

# --- Parse Inputs ---
def parse_input():
    sex_val = 1 if sex == "Male" else 0
    cp_val = int(cp[0])
    fbs_val = 1 if fbs.startswith("Yes") else 0
    restecg_val = int(restecg[0])
    exang_val = 1 if exang.startswith("Yes") else 0
    slope_val = int(slope[0])
    thal_val = int(thal[0])

    raw = {
        'age': age, 'trestbps': trestbps, 'chol': chol,
        'thalach': thalach, 'oldpeak': oldpeak,
        'sex': sex_val, 'cp': cp_val, 'fbs': fbs_val,
        'restecg': restecg_val, 'exang': exang_val,
        'slope': slope_val, 'ca': ca, 'thal': thal_val
    }

    df_input = pd.DataFrame([raw])
    df_input = pd.get_dummies(df_input, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

    # Align columns to match training data
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_cols]

    # Scale continuous features
    df_input[['age','trestbps','chol','thalach','oldpeak']] = scaler.transform(
        df_input[['age','trestbps','chol','thalach','oldpeak']]
    )

    return df_input

# --- Predict Button ---
if st.button(" Predict Risk", use_container_width=True):
    input_df = parse_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("###  High Risk : Heart Disease Likely")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
        st.markdown("""
        > This result suggests a higher likelihood of heart disease based on the entered values.  
        > Please consult a medical professional for proper diagnosis.
        """)
    else:
        st.success("### ✅ Low Risk — Heart Disease Unlikely")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")
        st.markdown("""
        > This result suggests a lower likelihood of heart disease based on the entered values.  
        > Regular check-ups are still recommended.
        """)

    st.markdown("---")
    st.caption("⚕️ This tool is for educational purposes only and is not a substitute for medical advice.")

# --- Sidebar Info ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** K-Nearest Neighbors (k=12)  
    **Dataset:** Cleveland Heart Disease Dataset  
    **Accuracy:** 84.48% (10-fold CV)  
    **Features:** 14 clinical indicators
    """)
    st.markdown("---")
    st.markdown("Built by **Kushagra Shekhar Kaushik**")
    st.markdown("[GitHub](https://github.com/kushagrakaush1k/heart-disease-prediction)")