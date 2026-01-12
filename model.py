import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
print("Loading data...")
data = pd.read_csv('data/creditcard.csv')

# Use only Amount and Time as features
X = data[['Time', 'Amount']]
y = data['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
print("Training model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as fraud_model.pkl")

