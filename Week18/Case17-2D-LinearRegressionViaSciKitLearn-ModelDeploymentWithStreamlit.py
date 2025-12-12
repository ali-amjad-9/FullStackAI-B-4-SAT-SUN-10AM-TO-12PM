import streamlit as st
import pickle
import numpy as np

# Load the model
with open("C:\\Users\\HP\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\predictive_maintenance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("Predictive Maintenance Dashboard")
st.write("Enter the student hours to predict marks:")

# Input fields for sensor data
hours = st.number_input("Students hours", value=5.5)

# Make prediction when button is clicked
if st.button("Predict"):
    features = np.array([hours]).reshape(1, -1)
    prediction = model.predict(features)
    st.write(prediction)