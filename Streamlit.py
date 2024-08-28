#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle


# In[4]:


# Load in the XGBoost model
filename = 'xgboost_model.pkl'  # Ensure this file is in the same directory or provide the full path
loaded_model = pickle.load(open(filename, 'rb'))

# Build a simple Streamlit app
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

# Predict button
predict_button = st.sidebar.button("Predict Fraud")

if predict_button:
    # Encode transaction type
    if transaction_type == 'PAYMENT':
        type_code = 0
    elif transaction_type == 'TRANSFER':
        type_code = 1
    elif transaction_type == 'CASH_OUT':
        type_code = 2
    elif transaction_type == 'CASH_IN':
        type_code = 3
    elif transaction_type == 'DEBIT':
        type_code = 4

    # Prepare the input for prediction
    input_data = np.array([[amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, type_code]])

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




