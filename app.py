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

# Create input fields for each feature
user_input = {}
for column in X.columns:
    if X[column].dtype == float:
        user_input[column] = st.number_input(f"Enter {column}:", value=float())
    elif X[column].dtype == int:
        user_input[column] = st.number_input(f"Enter {column}:", value=int())

# Prepare the input data as a dataframe
input_df = pd.DataFrame([user_input])

# Preprocess the input data
# Note: Make sure to preprocess the input data in the same way as the training data
input_scaled = scaler.transform(input_df)

# Make the prediction
prediction = mlp.predict_classes(input_scaled)
confidence = mlp.predict_proba(input_scaled) * 100

# Display the result
if prediction[0] == 0:
    st.write(f"Delayed Cerebral Ischemia is not predicted to occur. Confidence: {confidence[0][0]:.2f}%")
else:
    st.write(f"Delayed Cerebral Ischemia is predicted to occur. Confidence: {confidence[0][1]:.2f}%")
