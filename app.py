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

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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
        ('num', StandardScaler(), X.drop(columns=['Location of Aneurysm', 'Treatment Modality', 'Side']).columns),
        ('cat', OneHotEncoder(), ['Location of Aneurysm', 'Treatment Modality', 'Side'])
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
location_mapping = {
    "Anterior Communicating Artery": 1,
    "Middle Cerebral Artery": 2,
    "Anterior Cerebral Artery": 3,
    "Internal Carotid Artery": 4,
    "Posterior Communicating Artery": 5,
    "Basilar Artery": 6,
    "Posterior Inferior Cerebellar Artery": 7
}

side_mapping = {
    "Right": 0,
    "Left": 1,
}

treatment_mapping = {
    "Endovascular Coiling": 0,
    "Neurosurgical Clipping": 1,
}

label_mapping = {
    "WBCs on Admission": "WBCs on Admission",
    "HH Score": "HH Score",
    "mFisher Score": "mFisher Score",
    "Hypercholestorelemia": "Hypercholesterolemia",
    "Congestive Heart Failure": "Congestive Heart Failure",
    "Location of Aneurysm": "Location of Aneurysm",
    "Treatment Modality": "Treatment Modality",
    "EVD": "External Ventricular Drain",
    "VP Shunt": "VP Shunt",
    "TCD Vasospasm": "TCD Vasospasm",
    "Angiographic Vasospasm": "Angiographic Vasospasm",
    "Clinical Vasospasm": "Clinical Vasospasm",
    "WBCs": "White Blood Cells Count",
    "MCV": "MCV",
    "Red Cell Distribution Width": "Red Cell Distribution Width",
    "BUN": "BUN",
    "INR": "INR",
    "PTT": "PTT"
}

for column in X.columns:
    label = column.capitalize()

    if column in ['Nimodipine', 'Hypertension', 'Diabetes', 'Hypercholestorelemia',
                  'Congestive Heart Failure', 'Cancer', 'Smoking', 'Alcohol', 'Cocaine',
                  'EVD', 'VP Shunt', 'TCD Vasospasm', 'Angiographic Vasospasm', 'Clinical Vasospasm']:
        user_input[column] = st.selectbox(f"{label} (Yes/No):", options=["Yes", "No"])
        user_input[column] = 1 if user_input[column] == "Yes" else 0
    elif column == 'Location of Aneurysm':
        user_input[column] = st.selectbox(f"{label}:", options=list(location_mapping.keys()))
        user_input[column] = location_mapping[user_input[column]]
    elif column == 'Treatment Modality':
        user_input[column] = st.selectbox(f"{label}:", options=list(treatment_mapping.keys()))
        user_input[column] = treatment_mapping[user_input[column]]
    elif column in ['WBCs on Admission', 'Age', 'BMI', 'Size', 'WBCs', 'Neutrophils', 'Lymphocytes', 'Albumin', 'MCV', 'Platelets',
                    'Red Cell Distribution Width', 'Monocytes', 'BUN', 'Creatinine', 'INR', 'PTT']:
        user_input[column] = st.number_input(f"{label}:", step=None, format="%f")
    elif column in ['HH Score', 'mFisher Score']:
        user_input[column] = st.number_input(f"{label}:", min_value=0, step=1, format="%i")
    elif column == 'Side':
        user_input[column] = st.selectbox(f"{label}: ", options=list(side_mapping.keys()))
        user_input[column] = side_mapping[user_input[column]]

# Collect input data into a DataFrame
input_df = pd.DataFrame(user_input, index=[0])

# Preprocess the input data
input_preprocessed = preprocessor.transform(input_df)

# Make a prediction
prediction_proba = mlp.predict(input_preprocessed)
prediction = (prediction_proba > 0.5).astype(int)
confidence_percentage = round(prediction_proba[0][0] * 100, 2) if prediction[0][0] == 1 else round((1 - prediction_proba[0][0]) * 100, 2)

# Display the result
if st.button("Make Prediction"):
    result = "Delayed Cerebral Ischemia Is Predicted to Occur In This Patient" if prediction[0][0] == 1 else "Delayed Cerebral Ischemia Is NOT Predicted to Occur In This Patient"
    st.write(f"Prediction: **{result}**")
