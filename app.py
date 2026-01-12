import streamlit as st
import joblib
import numpy as np

# Set page title
st.title("Credit Card Fraud Detection")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'fraud_model.pkl' not found. Please run model.py first to train the model.")
        st.stop()

model = load_model()

# Input fields
st.header("Enter Transaction Details")

time = st.number_input("Time", min_value=0.0, value=0.0, step=1.0, help="Time of the transaction")
amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01, help="Transaction amount")

# Predict button
if st.button("Predict"):
    # Prepare input data
    transaction = np.array([[time, amount]])
    
    # Make prediction
    prediction = model.predict(transaction)[0]
    
    # Display result
    st.header("Prediction Result")
    
    if prediction == 1:
        st.error("FRAUD")
    else:
        st.success("NOT FRAUD")

