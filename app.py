import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
X_columns = joblib.load('X_columns.pkl')
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")
st.title("🫀 Heart Disease Prediction App")
st.write("Enter patient health details to predict heart disease likelihood.")

# ---- Input fields ----
age = st.number_input("Age", 18, 100, 50)
sex = st.radio("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1=normal, 2=fixed, 3=reversible)", [1, 2, 3])

# ---- Prepare DataFrame ----
sex_val = 1 if sex == "Male" else 0
new_df = pd.DataFrame([{
    'age': age, 'sex': sex_val, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}])

# One-hot encode like training
new_df = pd.get_dummies(new_df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
for col in X_columns:
    if col not in new_df.columns:
        new_df[col] = 0
new_df = new_df[X_columns]
new_df[columns_to_scale] = scaler.transform(new_df[columns_to_scale])

# ---- Prediction ----
if st.button("🔍 Predict"):
    pred = model.predict(new_df)[0]
    if pred == 1:
        st.error("🚨 The patient is **likely to have heart disease.**")
    else:
        st.success("✅ The patient is **likely healthy.**")
