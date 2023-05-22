import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Define the input features and their types
features = [
    ("WBCs on Admission", float),
    ("Age", int),
    ("BMI", float),
    ("Nimodipin", int),
    ("HH Score", int),
    ("mFisher Score", int),
    ("Hypertension", int),
    ("Diabetes", int),
    ("Hypercholestorelemia", int),
    ("Congestive Heart Failure", int),
    ("Cancer", int),
    ("Smoking", int),
    ("Cocaine", int),
    ("Location", int),
    ("Size", float),
    ("Side", int),
    ("Treatment Modality", int),
    ("EVD", int),
    ("VP Shunt", int),
    ("TCD Vasospasm", int),
    ("Angiographic Vasospasm", int),
    ("Clinical Vasospasm", int),
    ("WBCs", float),
    ("Neutrophils", float),
    ("Lymphocytes", float),
    ("Albumin", float),
    ("MCV", float),
    ("Platelets", float),
    ("Red Cell Distribution Width", float),
    ("Monocytes", float),
    ("BUN", float),
    ("Creatinine", float),
    ("INR", float),
    ("PTT", float)
]

st.title("Delayed Cerebral Ischemia Prediction")

# Create a dictionary to store user inputs
user_input = {}

# Create input fields for each feature
for feature, feature_type in features:
    user_input[feature] = st.number_input(f"Enter {feature}:", value=feature_type())

# Prepare the input data as a dataframe
input_df = pd.DataFrame([user_input])

# Preprocess the input data
# Note: Make sure to preprocess the input data in the same way as the training data
categorical_features = [
    "Nimodipine", "HH Score", "mFisher Score", "Hypertension", "Diabetes",
    "Hypercholestorelemia", "Congestive Heart Failure", "Cancer", "Smoking",
    "Alcohol", "Cocaine", "Location of Aneurysm", "Size", "Side",
    "Treatment Modality", "EVD", "VP Shunt", "TCD Vasospasm",
    "Angiographic Vasospasm", "Clinical Vasospasm"
]

input_df = pd.get_dummies(input_df, columns=categorical_features)

@st.experimental_singleton
def load_trained_model_and_scaler():
    if not os.path.isfile('trained_model.h5'):
        urllib.request.urlretrieve('https://github.com/drazzam/DCI_Predictor/raw/main/trained_model.h5', 'trained_model.h5')
    if not os.path.isfile('scaler.pkl'):
        urllib.request.urlretrieve('https://github.com/drazzam/DCI_Predictor/raw/main/scaler.pkl', 'scaler.pkl')

    mlp = load_model('trained_model.h5')
    scaler = joblib.load('scaler.pkl')
    return mlp, scaler

mlp, scaler = load_trained_model_and_scaler()

X = scaler.transform(input_df)

# Make the prediction
prediction = mlp.predict_classes(X)
confidence = mlp.predict_proba(X) * 100

# Display the result
if prediction[0] == 0:
    st.write(f"Delayed Cerebral Ischemia is not predicted to occur. Confidence: {confidence[0][0]:.2f}%")
else:
    st.write(f"Delayed Cerebral Ischemia is predicted to occur. Confidence: {confidence[0][1]:.2f}%")
