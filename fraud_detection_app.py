#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

# Load the trained model
model_path = 'models/xgboost_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the path.")
    st.stop()

# Set up the Streamlit app with a wider layout
st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 2])  # Adjust the width ratio (1:2 here)

with col1:
    # Left column: Data loading and preprocessing
    st.header('Data Loading and Pre-processing for Model Development')

    # Load your dataset
    data_path = 'Fraud.csv'
    try:
        df = pd.read_csv(data_path)
        st.write("Dataset loaded successfully.")
    except FileNotFoundError:
        st.error("Data file not found. Please check the path.")
        st.stop()

    # Correct column names to avoid duplicates and ensure consistency
    df = df.rename(columns={
        'oldbalanceOrg': 'oldbalanceOrg', 
        'newbalanceOrig': 'newbalanceOrig', 
        'oldbalanceDest': 'oldbalanceDest', 
        'newbalanceDest': 'newbalanceDest'
    })

    # Perform preprocessing and feature engineering as done during model training
    def preprocess_data(df):
        st.write("Preprocessing data...")

        # Remove any potential duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Add orig_diff if required columns are present
        if set(['oldbalanceOrg', 'newbalanceOrig', 'amount']).issubset(df.columns):
            df['orig_diff'] = (df['newbalanceOrig'] - df['oldbalanceOrg'] - df['amount']).abs().apply(lambda x: 1 if x > 0 else 0)
        else:
            st.error("Required columns for 'orig_diff' calculation are missing!")

        # Add dest_diff if required columns are present
        if set(['oldbalanceDest', 'newbalanceDest', 'amount']).issubset(df.columns):
            df['dest_diff'] = (df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']).abs().apply(lambda x: 1 if x > 0 else 0)
        else:
            st.error("Required columns for 'dest_diff' calculation are missing!")

        # Additional feature engineering
        if 'amount' in df.columns:
            df['surge'] = df['amount'].apply(lambda x: 1 if x > 450000 else 0)
        df['freq_dest'] = df['nameDest'].map(df['nameDest'].value_counts())
        df['freq_dest'] = df['freq_dest'].apply(lambda x: 1 if x > 20 else 0)
        df['merchant'] = df['nameDest'].apply(lambda x: 1 if x.startswith('M') else 0)

        if 'type' in df.columns:
            df = pd.concat([df, pd.get_dummies(df['type'], prefix='type_')], axis=1)
            df.drop(['type'], axis=1, inplace=True)
        
        df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, errors='ignore', inplace=True)

        st.write("Preprocessing complete. Shape after preprocessing:", df.shape)
        return df

    # Preprocess the dataset
    df_preprocessed = preprocess_data(df)

    # Display the preprocessed dataset
    with st.expander("Dataset with New Features"):
        st.write(df_preprocessed.head())  # Show the first few rows of the preprocessed dataframe

    # Option to download the preprocessed data
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_preprocessed)
    st.download_button(label="Download Preprocessed Data", data=csv, file_name='preprocessed_fraud_data.csv', mime='text/csv')

with col2:
    # Right column: Fraud Detection App
    st.title('Fraud Detection App')
    st.write("""
    This app predicts whether a financial transaction is fraudulent or not.
    """)

    # Form for user input
    with st.form(key='transaction_form'):
        step = st.number_input('Step (1 = 1 hour of time):', min_value=0)
        amount = st.number_input('Amount:', min_value=0.0, format='%f')
        oldbalanceOrg = st.number_input('Old Balance Original:', min_value=0.0, format='%f')
        newbalanceOrig = st.number_input('New Balance Original:', min_value=0.0, format='%f')
        oldbalanceDest = st.number_input('Old Balance Destination:', min_value=0.0, format='%f')
        newbalanceDest = st.number_input('New Balance Destination:', min_value=0.0, format='%f')
        orig_diff = st.number_input('Original Difference Flag (1 = Yes, 0 = No)', min_value=0, max_value=1)
        dest_diff = st.number_input('Destination Difference Flag (1 = Yes, 0 = No)', min_value=0, max_value=1)
        surge = st.number_input('Surge Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1)
        freq_dest = st.number_input('Frequency Destination Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1)
        merchant = st.number_input('Merchant Indicator (1 = Yes, 0 = No)', min_value=0, max_value=1)
        
        # Dropdown for transaction type
        transaction_type = st.selectbox(
            'Select Transaction Type:',
            options=['CASH_IN', 'CASH_OUT', 'PAYMENT', 'TRANSFER']
        )

        # Map dropdown to one-hot encoded columns
        type__CASH_IN = 1 if transaction_type == 'CASH_IN' else 0
        type__CASH_OUT = 1 if transaction_type == 'CASH_OUT' else 0
        type__PAYMENT = 1 if transaction_type == 'PAYMENT' else 0
        type__TRANSFER = 1 if transaction_type == 'TRANSFER' else 0

        # Button to submit the form
        submit_button = st.form_submit_button(label='Predict Fraud')

    # Only run prediction if the form is submitted
    if submit_button:
        # Collect input data into a DataFrame
        input_data = pd.DataFrame({
            'step': [step],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
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

        # Ensure input data has the same column order as the model expects
        input_data = input_data[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                                 'orig_diff', 'dest_diff', 'surge', 'freq_dest', 'merchant', 
                                 'type__CASH_IN', 'type__CASH_OUT', 'type__PAYMENT', 'type__TRANSFER']]

        # Make prediction using the input data
        try:
            prediction = model.predict(input_data)
            # Display the prediction result
            if prediction[0] == 1:
                st.error('The transaction is predicted to be fraudulent!')
            else:
                st.success('The transaction is predicted to be non-fraudulent.')
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Thank you message
    st.write('Thank you for using our app!')

