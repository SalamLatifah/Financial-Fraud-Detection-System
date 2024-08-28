#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
import xgboost as xgb 
import pickle


# In[10]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the model
with open('models/xgboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

st.set_page_config(layout="wide")
st.header('Financial Fraud Detection App')

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://img.freepik.com/premium-photo/wide-banner-with-many-random-square-hexagons-charcoal-dark-black-color_105589-1820.jpg" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

# Sidebar content
st.sidebar.subheader("Input Transaction Details")
st.sidebar.text("Please enter the details of the transaction below:")

# Input fields for transaction details
amount = st.sidebar.number_input('Transaction Amount')
oldbalanceOrig = st.sidebar.number_input('Original Balance Before Transaction')
newbalanceOrig = st.sidebar.number_input('New Balance After Transaction')
oldbalanceDest = st.sidebar.number_input('Original Balance of Receiver Before Transaction')
newbalanceDest = st.sidebar.number_input('New Balance of Receiver After Transaction')

# Transaction Type
transaction_type = st.sidebar.selectbox(
    'Transaction Type',
    ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT')
)

# Calculate engineered features based on input
orig_diff = 1 if newbalanceOrig != oldbalanceOrig - amount else 0
dest_diff = 1 if newbalanceDest != oldbalanceDest + amount else 0
surge = 1 if amount > 450000 else 0
freq_dest = 1  # This is a placeholder. In practice, this would be calculated based on historical data.
merchant = 1 if st.sidebar.checkbox('Is the receiver a merchant?') else 0

# One-hot encoding for transaction type
type_PAYMENT = 1 if transaction_type == 'PAYMENT' else 0
type_TRANSFER = 1 if transaction_type == 'TRANSFER' else 0
type_CASH_OUT = 1 if transaction_type == 'CASH_OUT' else 0
type_CASH_IN = 1 if transaction_type == 'CASH_IN' else 0
type_DEBIT = 1 if transaction_type == 'DEBIT' else 0

# Predict button
predict_button = st.sidebar.button("Predict Fraud")

if predict_button:
    # Prepare the input for prediction
    input_data = np.array([[amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, 
                            orig_diff, dest_diff, surge, freq_dest, merchant, 
                            type_PAYMENT, type_TRANSFER, type_CASH_OUT, type_CASH_IN, type_DEBIT]])

    # Predict using the loaded model
    result = loaded_model.predict(input_data)

    # Display the result
    if result[0] == 1:
        st.write("The transaction is predicted to be **fraudulent**.")
    else:
        st.write("The transaction is predicted to be **non-fraudulent**.")

    # Feedback option
    feedback = st.selectbox(
        'Is our prediction right or wrong?',
        ('Right', 'Wrong')
    )

    if feedback == 'Right':
        st.write('Thank you for your feedback!')
    else:
        st.write('Thank you for your feedback! We are constantly working to improve our model.')

st.write('Thank you for trying out our app!')
st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")


# In[ ]:




