import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import requests
from io import BytesIO

st.title("Delayed Cerebral Ischemia Prediction")

# Load the dataset from a CSV file
data_url = "https://github.com/drazzam/DCI_Predictor/raw/main/data.csv"
data_response = requests.get(data_url)
data = pd.read_csv(BytesIO(data_response.content))

# Preprocess the data
# Replace this with your own preprocessing steps based on the dataset
X = data.drop('dci', axis=1)
y = data['dci']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and compile the MLP model
mlp = Sequential()
mlp.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
mlp.add(Dense(32, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))
mlp.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Dictionary to map column names to user-friendly labels
col_labels = {
    'Nimodipine': 'Nimodipine',
    'Hypertension': 'Hypertension',
    'Diabetes': 'Diabetes',
    'Hypercholestorelemia': 'Hypercholesterolemia',
    'Congestive Heart Failure': 'Congestive Heart Failure',
    'Cancer': 'Cancer',
    'Smoking': 'Smoking',
    'Alcohol': 'Alcohol',
    'Cocaine': 'Cocaine',
    'Location of Aneurysm': 'Location of Aneurysm',
    'Treatment Modality': 'Treatment Modality',
    'EVD': 'EVD',
    'VP Shunt': 'VP Shunt',
    'TCD Vasospasm': 'TCD Vasospasm',
    'Angiographic Vasospasm': 'Angiographic Vasospasm',
    'Clinical Vasospasm': 'Clinical Vasospasm'
}

# Create input fields for each feature
user_input = {}
for column in X.columns:
    if column in col_labels:
        label = col_labels[column]
    else:
        label = column.capitalize()

    if column in ['Nimodipine', 'Hypertension', 'Diabetes', 'Hypercholestorelemia',
                  'Congestive Heart Failure', 'Cancer', 'Smoking', 'Alcohol', 'Cocaine',
                  'EVD', 'VP Shunt', 'TCD Vasospasm', 'Angiographic Vasospasm', 'Clinical Vasospasm']:
        user_input[column] = st.selectbox(f"{label} (Yes/No):", options=["Yes", "No"])
        user_input[column] = 1 if user_input[column] == "Yes" else 0
    elif column == 'Location of Aneurysm':
        user_input[column] = st.selectbox(f"{label}:", options=["Anterior Communicating Artery",
                                                                "Middle Cerebral Artery",
                                                                "Anterior Cerebral Artery",
                                                                "Internal Carotid Artery",
                                                                "Posterior Communicating Artery",
                                                                "Basilar Artery",
                                                                "Posterior Inferior Cerebellar Artery"])
        user_input[column] = {"Anterior Communicating Artery": 1,
                              "Middle Cerebral Artery": 2,
                              "Anterior Cerebral Artery": 3,
                              "Internal Carotid Artery": 4,
                              "Posterior Communicating Artery": 5,
                              "Basilar Artery": 6,
                              "Posterior Inferior Cerebellar Artery": 7}[user_input[column]]
    elif column == 'Treatment Modality':
        user_input[column] = st.selectbox(f"{label}:", options=["Endovascular Coiling",
                                                                "Neurosurgical Clipping"])
        user_input[column] = 0 if user_input[column] == "Endovascular Coiling" else 1

# Prepare the input data as adataframe
input_df = pd.DataFrame([user_input])

# Preprocess the input data
# Note: Make sure to preprocess the input data in the same way as the training data
input_scaled = scaler.transform(input_df)

# Make the prediction
prediction = (mlp.predict(input_scaled) > 0.5).astype("int32")
confidence = mlp.predict(input_scaled) * 100

# Display the result
if st.button("Make Prediction"):
    if prediction[0] == 0:
        st.write(f"Delayed Cerebral Ischemia is not predicted to occur. Confidence: {confidence[0][0]:.2f}%")
    else:
        st.write(f"Delayed Cerebral Ischemia is predicted to occur. Confidence: {confidence[0][1]:.2f}%")
