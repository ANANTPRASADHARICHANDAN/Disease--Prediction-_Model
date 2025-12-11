
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Multi-Disease Prediction", layout="wide")

try:
    heart_model = pickle.load(open('heart_model.pkl', 'rb'))
    diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
    breast_cancer_model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Run 'python train_models.py' first.")
    st.stop()

st.sidebar.title("Disease Prediction System")
selected = st.sidebar.selectbox("Select Disease", ["Heart Disease", "Diabetes", "Breast Cancer"])

if selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
        Smoking = st.selectbox("Smoking", ["Yes", "No"])
        AlcoholDrinking = st.selectbox("Alcohol", ["Yes", "No"])
        Stroke = st.selectbox("Stroke", ["Yes", "No"])
        PhysicalHealth = st.number_input("Physical Health (Days)", 0, 30, 0)
        MentalHealth = st.number_input("Mental Health (Days)", 0, 30, 0)
    with col2:
        DiffWalking = st.selectbox("DiffWalking", ["Yes", "No"])
        Sex = st.selectbox("Sex", ["Male", "Female"])
        AgeCategory = st.selectbox("Age", ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'])
        Race = st.selectbox("Race", ["White", "Black", "Asian", "Other", "Hispanic"])
        Diabetic = st.selectbox("Diabetic", ["Yes", "No"])
    with col3:
        PhysicalActivity = st.selectbox("PhysicalActivity", ["Yes", "No"])
        GenHealth = st.selectbox("GenHealth", ["Excellent", "Very good", "Good", "Fair", "Poor"])
        SleepTime = st.number_input("SleepTime", 0.0, 24.0, 7.0)
        Asthma = st.selectbox("Asthma", ["Yes", "No"])
        KidneyDisease = st.selectbox("KidneyDisease", ["Yes", "No"])
        SkinCancer = st.selectbox("SkinCancer", ["Yes", "No"])

    if st.button("Predict Heart Disease"):
        input_data = pd.DataFrame({
            'BMI': [BMI], 'PhysicalHealth': [PhysicalHealth], 'MentalHealth': [MentalHealth], 'SleepTime': [SleepTime],
            'Smoking': [Smoking], 'AlcoholDrinking': [AlcoholDrinking], 'Stroke': [Stroke], 'DiffWalking': [DiffWalking],
            'Sex': [Sex], 'AgeCategory': [AgeCategory], 'Race': [Race], 'Diabetic': [Diabetic], 
            'PhysicalActivity': [PhysicalActivity], 'GenHealth': [GenHealth], 'Asthma': [Asthma], 
            'KidneyDisease': [KidneyDisease], 'SkinCancer': [SkinCancer]
        })
        st.write("Result:", "High Risk" if heart_model.predict(input_data)[0] == 1 else "Healthy")

elif selected == "Diabetes":
    st.title("Diabetes Prediction")
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        Glucose = st.number_input("Glucose", 0, 200, 100)
        BloodPressure = st.number_input("BP", 0, 150, 70)
        SkinThickness = st.number_input("SkinThickness", 0, 100, 20)
    with col2:
        Insulin = st.number_input("Insulin", 0, 900, 79)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        DPF = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 0, 100, 30)
    
    if st.button("Predict Diabetes"):
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]], 
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        st.write("Result:", "Diabetic" if diabetes_model.predict(input_data)[0] == 1 else "Not Diabetic")

elif selected == "Breast Cancer":
    st.title("Breast Cancer Prediction")
    # Quick input generation for 30 features
    features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
    input_vals = []
    col1, col2, col3 = st.columns(3)
    for i, f in enumerate(features):
        with col1: m = st.number_input(f"Mean {f}", 0.0)
        with col2: e = st.number_input(f"{f} Error", 0.0)
        with col3: w = st.number_input(f"Worst {f}", 0.0)
        input_vals.extend([m, e, w])
    
    # Sort inputs to match model order (mean1...mean10, error1...error10, worst1...worst10)
    # The loop above collects them interleaved (m,e,w, m,e,w), so we need to reorganize or just input them correctly
    # Simpler approach for this quick fix: Just take the list as is (Caution: Real model needs exact order)
    # For exact correctness, we should collect them into 3 separate lists
    
    if st.button("Predict Cancer"):
         # Re-organize to [means..., errors..., worsts...]
        means = input_vals[0::3]
        errors = input_vals[1::3]
        worsts = input_vals[2::3]
        final_input = means + errors + worsts
        
        # Column names are required for the scaler
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        input_df = pd.DataFrame([final_input], columns=data.feature_names)
        
        st.write("Result:", "Malignant" if breast_cancer_model.predict(input_df)[0] == 0 else "Benign")
