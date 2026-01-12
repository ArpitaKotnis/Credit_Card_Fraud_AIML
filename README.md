# Credit Card Fraud Detection

A simple Python project that detects fraudulent credit card transactions using machine learning.

## What does this project do?

This project uses a Logistic Regression model to predict whether a credit card transaction is fraudulent or not. It uses only two features: the time of the transaction and the transaction amount.

## How to use

### Step 1: Install dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 2: Add your data

Place your credit card dataset as `data/creditcard.csv`. The CSV file should have columns including:
- `Time`: Time of the transaction
- `Amount`: Transaction amount
- `Class`: 0 for normal transactions, 1 for fraudulent transactions

### Step 3: Train the model

Run the training script to create the model:

```bash
python model.py
```

This will:
- Load the data from `data/creditcard.csv`
- Train a Logistic Regression model
- Save the model as `fraud_model.pkl`

### Step 4: Make predictions

Run the prediction script to test a transaction:

```bash
python predict.py
```

This will load the saved model and predict whether a sample transaction is fraudulent or not.

## Files

- `model.py`: Trains and saves the fraud detection model
- `predict.py`: Loads the model and makes predictions
- `requirements.txt`: List of required Python packages
- `data/creditcard.csv`: Your credit card transaction data (you need to provide this)

## Note

Make sure you have the credit card dataset file (`creditcard.csv`) in the `data/` folder before running `model.py`.

<img width="872" height="509" alt="arpita_creditcard" src="https://github.com/user-attachments/assets/45ecdfe6-7270-4a77-a47e-53feb2a13fdc" />




