import streamlit as st
import pandas as pd
import joblib


# Page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# Load model
model = joblib.load("diabetes_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# Title
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient information below "
    "to predict the diabetes outcome."
)


# Input fields
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=120,
    step=1
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70,
    step=1
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=1000,
    value=80,
    step=1
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.01
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)


# Prediction button
if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # Result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Diabetes (1)")
    else:
        st.success("Prediction: No Diabetes (0)")


    st.info(
        f"Diabetes Probability: {probability * 100:.2f}%"
    )


st.caption(
    "This application is for educational purposes only "
    "and is not a medical diagnosis."
)