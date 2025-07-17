import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
df = pd.read_csv('Fraud Detection in Financial Transactions - DataSet.csv')

# Data Exploration and Preprocessing
print("Initial Data Overview:")
print(df.head())
print("\nData Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Feature Engineering
# Convert TransactionDate to datetime and extract features
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionDayOfWeek'] = df['TransactionDate'].dt.dayofweek

# Convert PreviousTransactionDate to datetime and calculate time since last transaction
df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
df['TimeSinceLastTransaction'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 3600  # in hours

# Calculate transaction amount as percentage of account balance
df['AmountToBalanceRatio'] = df['TransactionAmount'] / df['AccountBalance']

# Encode categorical variables
categorical_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Extract first 3 octets of IP address as a feature
df['IPPrefix'] = df['IP Address'].apply(lambda x: '.'.join(x.split('.')[:3]))

# Feature selection
features = ['TransactionAmount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 
            'TransactionDayOfWeek', 'TimeSinceLastTransaction', 'AmountToBalanceRatio',
            'TransactionType', 'Location', 'Channel', 'CustomerOccupation', 
            'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance']

X = df[features]
y = df['LoginAttempts'] > 1  # Using LoginAttempts > 1 as a proxy for suspicious activity (for demonstration)

# Handle missing values if any
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Function to detect potential fraud in new transactions
def detect_fraud(new_transaction):
    # Preprocess the new transaction similar to training data
    new_transaction['TransactionDate'] = pd.to_datetime(new_transaction['TransactionDate'])
    new_transaction['TransactionHour'] = new_transaction['TransactionDate'].dt.hour
    new_transaction['TransactionDay'] = new_transaction['TransactionDate'].dt.day
    new_transaction['TransactionMonth'] = new_transaction['TransactionDate'].dt.month
    new_transaction['TransactionDayOfWeek'] = new_transaction['TransactionDate'].dt.dayofweek
    
    new_transaction['PreviousTransactionDate'] = pd.to_datetime(new_transaction['PreviousTransactionDate'])
    new_transaction['TimeSinceLastTransaction'] = (new_transaction['TransactionDate'] - new_transaction['PreviousTransactionDate']).dt.total_seconds() / 3600
    
    new_transaction['AmountToBalanceRatio'] = new_transaction['TransactionAmount'] / new_transaction['AccountBalance']
    
    for col in categorical_cols:
        new_transaction[col] = label_encoders[col].transform([str(new_transaction[col])])[0]
    
    # Select features and scale
    new_X = new_transaction[features]
    new_X = pd.DataFrame(imputer.transform(new_X), columns=features)
    new_X_scaled = scaler.transform(new_X)
    
    # Predict
    prediction = model.predict(new_X_scaled)
    probability = model.predict_proba(new_X_scaled)[:, 1]
    
    return prediction[0], probability[0]

# Example usage with a new transaction
new_transaction = {
    'TransactionID': 'TX999999',
    'AccountID': 'AC99999',
    'TransactionAmount': 1500.00,
    'TransactionDate': '2023-11-04 03:15:00',  # Unusual hour
    'TransactionType': 'Debit',
    'Location': 'New York',
    'DeviceID': 'D999999',
    'IP Address': '192.168.1.1',
    'MerchantID': 'M999',
    'Channel': 'Online',
    'CustomerAge': 35,
    'CustomerOccupation': 'Engineer',
    'TransactionDuration': 30,
    'LoginAttempts': 5,  # Multiple login attempts
    'AccountBalance': 2000.00,
    'PreviousTransactionDate': '2023-11-04 01:00:00'  # Recent transaction
}

is_fraud, fraud_probability = detect_fraud(new_transaction)
print(f"\nFraud Detection Result for New Transaction:")
print(f"Is Fraudulent: {is_fraud}")
print(f"Fraud Probability: {fraud_probability:.2%}")