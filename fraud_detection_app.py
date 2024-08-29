#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle


# In[7]:


# Load the trained model
model_path = 'models/xgboost_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit app
st.title('Fraud Detection App')
st.write("""
This app stands as a prototype where it should be able to predicts whether a financial transaction is fraudulent or not.
This app was build on XGBoost model.
""")

# Create a form to input new transaction data
st.header('Enter Transaction Data')

# Input fields for transaction data with explanations
step = st.number_input('Step (1 = 1 hour of time):', min_value=0, help="The step represents time (1 step = 1 hour).")
amount = st.number_input('Amount:', min_value=0.0, format='%f', help="Enter the transaction amount.")
oldbalanceOrig = st.number_input('Old Balance Original:', min_value=0.0, format='%f', help="The initial balance before the transaction.")
newbalanceOrig = st.number_input('New Balance Original:', min_value=0.0, format='%f', help="The new balance after the transaction.")
oldbalanceDest = st.number_input('Old Balance Destination:', min_value=0.0, format='%f', help="The initial balance of the destination account before the transaction.")
newbalanceDest = st.number_input('New Balance Destination:', min_value=0.0, format='%f', help="The new balance of the destination account after the transaction.")
orig_diff = st.number_input('Original Difference Flag (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicates a discrepancy in the sender's account after the transaction.")
dest_diff = st.number_input('Destination Difference Flag (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicates a discrepancy in the destination account after the transaction.")
surge = st.number_input('Surge Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicates whether the transaction amount is unusually large.")
freq_dest = st.number_input('Frequency Destination Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicates whether the destination account frequently receives transactions.")
merchant = st.number_input('Merchant Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicates whether the destination account is a merchant.")
type__CASH_IN = st.number_input('Type: CASH_IN (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicate '1' if this is a CASH_IN transaction.")
type__CASH_OUT = st.number_input('Type: CASH_OUT (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicate '1' if this is a CASH_OUT transaction.")
type__PAYMENT = st.number_input('Type: PAYMENT (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicate '1' if this is a PAYMENT transaction.")
type__TRANSFER = st.number_input('Type: TRANSFER (1 = Yes, 0 = No)', min_value=0, max_value=1, help="Indicate '1' if this is a TRANSFER transaction.")

# Collect input data into a DataFrame with the correct feature names
input_data = pd.DataFrame({
    'step': [step],
    'amount': [amount],
    'oldbalanceOrg': [oldbalanceOrig],
    'newbalanceOrig': [newbalanceOrig],
    'oldbalanceDest': [oldbalanceDest],
    'newbalanceDest': [newbalanceDest],
    'orig_diff': [orig_diff],
    'dest_diff': [dest_diff],
    'surge': [surge],
    'freq_dest': [freq_dest],
    'merchant': [merchant],
    'type__CASH_IN': [type__CASH_IN],
    'type__CASH_OUT': [type__CASH_OUT],
    'type__PAYMENT': [type__PAYMENT],
    'type__TRANSFER': [type__TRANSFER]
})

# Button to make prediction
if st.button('Predict Fraud'):
    # Make prediction using the input data
    prediction = model.predict(input_data)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.error('The transaction is predicted to be fraudulent!')
    else:
        st.success('The transaction is predicted to be non-fraudulent.')


# In[ ]:




