"""
Module for data preprocessing, feature engineering, and model training on Lending Club loan data.

This script:
- Cleans and encodes categorical variables
- Splits data into training and testing sets
- Applies standard scaling
- Trains and evaluates Random Forest, Logistic Regression, and MLP classifiers

Authors : Luc Kouassi, Long Hoang, Caleb Chidi
Date of creation : June 2025

"""

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import hashlib
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random

# Fix random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ================================
# Load dataset
# ================================
lending_club = pd.read_csv('loans_full_schema.csv')

# Create initial report
profile = ProfileReport(lending_club, title="CS Lending Club Report")
profile.to_file("CS Lending Club Report.html")

# ================================
# Data Preprocessing and Encoding
# ================================

def hash_encode(val: str, num_buckets: int = 4742) -> int:
    """
    Hash-encodes a string value to a bucket.

    Parameters
    ----------
    val : str
        The string value to hash.
    num_buckets : int, optional
        The number of hash buckets. Default is 4742.

    Returns
    -------
    int
        Hash bucket index.
    """
    return int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16) % num_buckets

def hash_encode_state(val: str, num_buckets: int = 50) -> int:
    """
    Hash-encodes a state string value to a bucket.

    Parameters
    ----------
    val : str
        The state value to hash.
    num_buckets : int, optional
        The number of hash buckets. Default is 50.

    Returns
    -------
    int
        Hash bucket index.
    """
    return int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16) % num_buckets

# Fill missing 'emp_title' with 'others' and apply hashing
lending_club['emp_title'] = lending_club['emp_title'].fillna('others')
lending_club['emp_title_hashed'] = lending_club['emp_title'].apply(lambda x: hash_encode(str(x)))

# Fill missing 'emp_length' with median
median_emp_length = lending_club['emp_length'].median()
lending_club['emp_length'] = lending_club['emp_length'].fillna(median_emp_length)

# Hash-encode 'state'
lending_club['state_hashed'] = lending_club['state'].apply(lambda x: hash_encode_state(str(x)))

# Encode 'homeownership'
homeownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2}
lending_club['homeownership_encoded'] = lending_club['homeownership'].map(homeownership_map)

# Encode 'verified_income'
verified_income_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
lending_club['verified_income_encoded'] = lending_club['verified_income'].map(verified_income_map)

# Fill missing and encode 'verification_income_joint'
lending_club['verification_income_joint'] = lending_club.apply(
    lambda row: 'no_joint' if pd.isna(row['verification_income_joint']) and row['application_type'] == 'individual'
    else ('Not Verified' if pd.isna(row['verification_income_joint']) and row['application_type'] == 'joint'
          else row['verification_income_joint']),
    axis=1
)
verification_income_joint_map = {'no_joint': 0, 'Not Verified': 1, 'Source Verified': 2, 'Verified': 3}
lending_club['verification_income_joint_encoded'] = lending_club['verification_income_joint'].map(verification_income_joint_map)

# Encode 'application_type'
application_type_map = {'individual': 0, 'joint': 1}
lending_club['application_type_encoded'] = lending_club['application_type'].map(application_type_map)

# Binary-encode 'loan_status'
loan_status_map = {
    'Fully Paid': 0,
    'Current': 0,
    'In Grace Period': 1,
    'Late (16-30 days)': 1,
    'Late (31-120 days)': 1,
    'Charged Off': 1
}
lending_club['loan_status_binary'] = lending_club['loan_status'].map(loan_status_map)

# Fill missing values
lending_club['debt_to_income_joint'] = lending_club['debt_to_income_joint'].fillna(0)
mode_months_since_last_delinq = lending_club['months_since_last_delinq'].mode().iloc[0]
lending_club['months_since_last_delinq'] = lending_club.apply(
    lambda row: 0 if pd.isna(row['months_since_last_delinq']) and row['account_never_delinq_percent'] == 100
    else (mode_months_since_last_delinq if pd.isna(row['months_since_last_delinq']) and row['account_never_delinq_percent'] < 100
          else row['months_since_last_delinq']),
    axis=1
)

# Convert and fill numerical columns
lending_club['debt_to_income'] = pd.to_numeric(lending_club['debt_to_income'], errors='coerce').fillna(lending_club['debt_to_income'].median())
lending_club['annual_income_joint'] = pd.to_numeric(lending_club['annual_income_joint'], errors='coerce').fillna(0)
lending_club['months_since_last_credit_inquiry'] = pd.to_numeric(lending_club['months_since_last_credit_inquiry'], errors='coerce').fillna(
    lending_club['months_since_last_credit_inquiry'].median())
lending_club['num_accounts_120d_past_due'] = pd.to_numeric(lending_club['num_accounts_120d_past_due'], errors='coerce').fillna(1)

# Encode 'sub_grade' ordinally
subgrades_order = ['A1', 'A2', 'A3', 'A4', 'A5',
                   'B1', 'B2', 'B3', 'B4', 'B5',
                   'C1', 'C2', 'C3', 'C4', 'C5',
                   'D1', 'D2', 'D3', 'D4', 'D5',
                   'E1', 'E2', 'E3', 'E4', 'E5',
                   'F1', 'F2', 'F3', 'F4', 'F5',
                   'G1', 'G4']
encoder = ce.OrdinalEncoder(
    cols=['sub_grade'],
    mapping=[{'col': 'sub_grade', 'mapping': {val: i for i, val in enumerate(subgrades_order)}}]
)
lending_club = encoder.fit_transform(lending_club)

# Encode 'issue_month'
months_mapping = {'Jan-2018': 1, 'Feb-2018': 2, 'Mar-2018': 3}
lending_club['issue_month_encoded'] = lending_club['issue_month'].map(months_mapping)

# Fill missing values
lending_club['months_since_90d_late'] = lending_club['months_since_90d_late'].fillna(0)

# Encode 'loan_purpose' using frequency
loan_purpose_freq = lending_club['loan_purpose'].value_counts(normalize=True)
lending_club['loan_purpose_freq_encoded'] = lending_club['loan_purpose'].map(loan_purpose_freq)

# Encode additional categorical variables
term_mapping = {36: 0, 60: 1}
lending_club['term_encoded'] = lending_club['term'].map(term_mapping)
grade_mapping = {'G': 6, 'F': 5, 'E': 4, 'D': 3, 'C': 2, 'B': 1, 'A': 0}
lending_club['grade_encoded'] = lending_club['grade'].map(grade_mapping)
initial_listing_mapping = {'whole': 0, 'fractional': 1}
lending_club['initial_listing_status_encoded'] = lending_club['initial_listing_status'].map(initial_listing_mapping)
disbursement_mapping = {'Cash': 0, 'DirectPay': 1}
lending_club['disbursement_method_encoded'] = lending_club['disbursement_method'].map(disbursement_mapping)

# ================================
# Feature Selection & Train-Test Split
# ================================

# Exclude cleaned columns from features
feature_columns = lending_club.columns.difference([
    'emp_title', 'state', 'homeownership', 'issue_month',
    'verified_income', 'verification_income_joint', 'application_type',
    'loan_status', 'loan_status_binary', 'loan_purpose', 'term',
    'grade', 'initial_listing_status', 'disbursement_method'
])

X = lending_club[feature_columns]
y = lending_club['loan_status_binary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Model Training and Evaluation
# ================================

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print accuracy, classification report, and confusion matrix.

    Parameters
    ----------
    name : str
        Name of the model.
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    print(f"\n{name} Accuracy:", accuracy_score(y_true, y_pred))
    print(f"\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ----- Random Forest -----
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ----- Logistic Regression -----
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
evaluate_model("Logistic Regression", y_test, y_pred_lr)

# ----- MLP (Neural Network) -----
mlp_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)

y_probs_mlp = mlp_model.predict(X_test_scaled).ravel()
y_pred_mlp = (y_probs_mlp > 0.5).astype(int)
evaluate_model("MLP", y_test, y_pred_mlp)
