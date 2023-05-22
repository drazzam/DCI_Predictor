import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
X = data.drop('dci', axis=1)
y = data['dci']

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns),
        ('cat', OneHotEncoder(), ['Location of Aneurysm', 'Treatment Modality'])
    ])

# Apply the column transformer to the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Define and compile the MLP model
mlp = Sequential()
mlp.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
mlp.add(Dense(32, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))
mlp.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
mlp.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Create input fields for each feature
user_input = {}
for column in X.columns:
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
    elif column == 'Treatment Modality':
        user_input[column] = st.selectbox(f"{label}:", options=["Endovascular Coiling",
                                                                "Neurosurgical Clipping"])
    elif column in ['WBCs on Admission', 'Age', 'BMI', 'Size', 'WBCs', 'Neutrophils', 'Lymphocytes', 'Albumin', 'MCV', 'Platelets',
                    'Red Cell Distribution Width', 'Monocytes', 'BUN', 'Creatinine', 'INR', 'PTT']:
        user_input[column] = st.number_input(f"{label}:", step=None, format="%f")
    elif column in ['HH Score', 'mFisher Score']:
        user_input[column] = st.number_input(f"{label}:", min_value=0, step=1, format="%i")
    elif column == 'Side':
        user_input[column] = st.selectbox(f"{label}":, options= ["Right", "Left"])

# Prepare the input data as a DataFrame
input_df = pd.DataFrame([user_input])

# Preprocess the input data
input_preprocessed = preprocessor.transform(input_df)

# Make the prediction
prediction = (mlp.predict(input_preprocessed) > 0.5).astype("int32")
confidence = mlp.predict(input_preprocessed) * 100

# Display the result
if st.button("Make Prediction"):
    if prediction[0] == 0:
        st.write(f"Delayed Cerebral Ischemia is not predicted to occur. Confidence: {confidence[0][0]:.2f}%")
    else:
        st.write(f"Delayed Cerebral Ischemia is predicted to occur. Confidence: {confidence[0][0]:.2f}%")
