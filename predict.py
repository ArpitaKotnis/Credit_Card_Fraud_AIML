import joblib
import numpy as np

# Load the trained model
print("Loading model...")
model = joblib.load('fraud_model.pkl')

# Sample transaction: Time and Amount
# You can change these values to test different transactions
sample_transaction = np.array([[100000, 100.50]])

# Make prediction
prediction = model.predict(sample_transaction)

# Print result
if prediction[0] == 1:
    print("FraUD")
else:
    print("NOT FraUD")

