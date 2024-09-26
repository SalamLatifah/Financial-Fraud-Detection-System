---

# Financial Fraud Detection System

This project involves building a robust model to detect fraudulent financial transactions using machine learning techniques. The model aims to enhance early detection of fraud, thereby minimizing financial losses and protecting customers.

## Overview

Financial fraud detection is a critical area in the financial sector. Early detection can save organizations significant losses and protect customer accounts from fraudulent activities. This project utilizes three machine learning algorithms to build a predictive model that identifies potentially fraudulent transactions.

## Project Structure

- **Data**: The dataset used includes various features related to transactions, such as transaction amount, balances before and after transactions, and transaction types.
- **Preprocessing**: Data preprocessing steps include handling missing values, encoding categorical variables, and standardizing numerical features.
- **Models**: Multiple models were trained and evaluated, including Logistic Regression, Support Vector Machine, and XGBoost. The XGBoost model was selected for deployment due to its high performance, although some overfitting was observed.
- **Deployment**: The model is deployed using Streamlit, providing a user-friendly web interface to predict fraudulent transactions based on input data.

## Features

- **Real-time Fraud Detection**: Input transaction details to predict whether a transaction is fraudulent.
- **User Interface**: Built using Streamlit for easy interaction.
- **Code and Documentation**: Organized and well-documented for easy understanding and further development.

## Getting Started

### Prerequisites

- Python 3.9
- pip for package management

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/LT2-SALAM/Fraud-detection-system-using-maching-learning-models.git
    ```

2. Navigate to the project directory:

    ```bash
    cd financial-fraud-detection-system
    ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Download the Dataset

Before running the code, ensure you have downloaded the dataset required for this project:

1. Download the dataset file from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).

2. Save the downloaded file as `Fraud.csv` directly within the project directory. The final path should look like `financial-fraud-detection-system/Fraud.csv`.

3. Ensure the dataset file is correctly named and placed so the application can access it.

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run fraud_detection_app.py
    ```

2. Follow the prompts to input transaction data and receive fraud detection predictions.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature additions or bug fixes.

## Acknowledgments

Special thanks to everyone who contributed to this project.
