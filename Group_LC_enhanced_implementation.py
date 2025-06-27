"""
Module for advanced preprocessing, resampling, feature selection, and model evaluation
using cross-validation on Lending Club loan data.

This script:
- Applies hash and categorical encoding for feature engineering
- Handles class imbalance using SMOTETomek resampling
- Performs feature selection using logistic regression with balanced class weights
- Trains and evaluates Bagged Logistic Regression, Random Forest, and MLP classifiers
- Optimizes classification thresholds per model for improved minority class detection
- Aggregates performance metrics across folds and visualizes confusion matrices

Authors : Luc Kouassi, Long Hoang, Caleb Chidi
Date of creation : June 2025

"""

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import hashlib
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

# Fix random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load dataset
lending_club = pd.read_csv('loans_full_schema.csv')

# Create initial report
profile = ProfileReport(lending_club, title="CS Lending Club Report")
profile.to_file("CS Lending Club Report.html")

# --- Data Cleaning and Feature Engineering ---
lending_club['emp_title'] = lending_club['emp_title'].fillna('others')

def hash_encode(val, num_buckets=4742):
    """
    Hash-encode a string value into an integer bucket.

    Parameters
    ----------
    val : str
        String value to encode.
    num_buckets : int, optional
        Number of hash buckets to distribute values into, by default 4742.

    Returns
    -------
    int
        Hashed value as an integer.
    """
    return int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16) % num_buckets

lending_club['emp_title_hashed'] = lending_club['emp_title'].apply(lambda x: hash_encode(str(x)))

median_emp_length = lending_club['emp_length'].median()
lending_club['emp_length'] = lending_club['emp_length'].fillna(median_emp_length)

def hash_encode_state(val, num_buckets=50):
    """
    Hash-encode the 'state' feature into an integer bucket.

    Parameters
    ----------
    val : str
        State value to encode.
    num_buckets : int, optional
        Number of hash buckets, by default 50.

    Returns
    -------
    int
        Encoded state as an integer.
    """
    return int(hashlib.sha256(val.encode('utf-8')).hexdigest(), 16) % num_buckets

lending_club['state_hashed'] = lending_club['state'].apply(lambda x: hash_encode_state(str(x)))

homeownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2}
lending_club['homeownership_encoded'] = lending_club['homeownership'].map(homeownership_map)

verified_income_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
lending_club['verified_income_encoded'] = lending_club['verified_income'].map(verified_income_map)

lending_club['verification_income_joint'] = lending_club.apply(
    lambda row: 'no_joint' if pd.isna(row['verification_income_joint']) and row['application_type'] == 'individual'
    else ('Not Verified' if pd.isna(row['verification_income_joint']) and row['application_type'] == 'joint'
          else row['verification_income_joint']),
    axis=1
)
verification_income_joint_map = {'no_joint': 0, 'Not Verified': 1, 'Source Verified': 2, 'Verified': 3}
lending_club['verification_income_joint_encoded'] = lending_club['verification_income_joint'].map(verification_income_joint_map)

application_type_map = {'individual': 0, 'joint': 1}
lending_club['application_type_encoded'] = lending_club['application_type'].map(application_type_map)

loan_status_map = {
    'Fully Paid': 0, 'Current': 0, 'In Grace Period': 1,
    'Late (16-30 days)': 1, 'Late (31-120 days)': 1, 'Charged Off': 1
}
lending_club['loan_status_binary'] = lending_club['loan_status'].map(loan_status_map)

lending_club['debt_to_income_joint'] = lending_club['debt_to_income_joint'].fillna(0)

mode_months_since_last_delinq = lending_club['months_since_last_delinq'].mode().iloc[0]
lending_club['months_since_last_delinq'] = lending_club.apply(
    lambda row: 0 if pd.isna(row['months_since_last_delinq']) and row['account_never_delinq_percent'] == 100
    else (mode_months_since_last_delinq if pd.isna(row['months_since_last_delinq']) and row['account_never_delinq_percent'] < 100
          else row['months_since_last_delinq']),
    axis=1
)

lending_club['debt_to_income'] = pd.to_numeric(lending_club['debt_to_income'], errors='coerce')
lending_club['debt_to_income'] = lending_club['debt_to_income'].fillna(lending_club['debt_to_income'].median())

lending_club['annual_income_joint'] = pd.to_numeric(lending_club['annual_income_joint'], errors='coerce').fillna(0)
lending_club['months_since_last_credit_inquiry'] = pd.to_numeric(lending_club['months_since_last_credit_inquiry'], errors='coerce').fillna(
    lending_club['months_since_last_credit_inquiry'].median())
lending_club['num_accounts_120d_past_due'] = pd.to_numeric(lending_club['num_accounts_120d_past_due'], errors='coerce').fillna(1)

subgrades_order = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                   'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
                   'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
                   'G1', 'G4']
encoder = ce.OrdinalEncoder(
    cols=['sub_grade'],
    mapping=[{'col': 'sub_grade', 'mapping': {val: i for i, val in enumerate(subgrades_order)}}]
)
lending_club = encoder.fit_transform(lending_club)

months_mapping = {'Jan-2018': 1, 'Feb-2018': 2, 'Mar-2018': 3}
lending_club['issue_month_encoded'] = lending_club['issue_month'].map(months_mapping)
lending_club['months_since_90d_late'] = lending_club['months_since_90d_late'].fillna(0)
loan_purpose_freq = lending_club['loan_purpose'].value_counts(normalize=True)
lending_club['loan_purpose_freq_encoded'] = lending_club['loan_purpose'].map(loan_purpose_freq)
term_mapping = {36: 0, 60: 1}
lending_club['term_encoded'] = lending_club['term'].map(term_mapping)
grade_mapping = {'G': 6, 'F': 5, 'E': 4, 'D': 3, 'C': 2, 'B': 1, 'A': 0}
lending_club['grade_encoded'] = lending_club['grade'].map(grade_mapping)
initial_listing_mapping = {'whole': 0, 'fractional': 1}
lending_club['initial_listing_status_encoded'] = lending_club['initial_listing_status'].map(initial_listing_mapping)
disbursement_mapping = {'Cash': 0, 'DirectPay': 1}
lending_club['disbursement_method_encoded'] = lending_club['disbursement_method'].map(disbursement_mapping)

# Define feature columns (excluding original categorical and target columns)
feature_columns = lending_club.columns.difference([
    'emp_title', 'state', 'homeownership', 'issue_month',
    'verified_income', 'verification_income_joint', 'application_type',
    'loan_status', 'loan_status_binary', 'loan_purpose', 'term',
    'grade', 'initial_listing_status', 'disbursement_method',
])
X = lending_club[feature_columns]
y = lending_club['loan_status_binary']

# --- Helper Functions ---

def optimize_threshold(y_true, y_probs):
    """
    Optimize classification threshold based on maximizing F1-score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0 or 1).
    y_probs : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Optimal threshold for classification.
    """
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (y_probs >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def extract_metrics(report_dict, metrics_dict):
    """
    Extract classification metrics from a dictionary and store in a metrics dict.

    Parameters
    ----------
    report_dict : dict
        Classification report dictionary.
    metrics_dict : dict
        Dictionary to store the metrics.
    """
    metrics_dict['accuracy'].append(report_dict['accuracy'])
    metrics_dict['precision_0'].append(report_dict['0']['precision'])
    metrics_dict['recall_0'].append(report_dict['0']['recall'])
    metrics_dict['f1_0'].append(report_dict['0']['f1-score'])
    metrics_dict['precision_1'].append(report_dict['1']['precision'])
    metrics_dict['recall_1'].append(report_dict['1']['recall'])
    metrics_dict['f1_1'].append(report_dict['1']['f1-score'])

# --- Cross-Validation Setup ---
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

# Initialize metrics dictionaries to store performance metrics for each model
lr_metrics = {'accuracy': [], 'precision_0': [], 'recall_0': [], 'f1_0': [],
              'precision_1': [], 'recall_1': [], 'f1_1': []}
rf_metrics = {'accuracy': [], 'precision_0': [], 'recall_0': [], 'f1_0': [],
              'precision_1': [], 'recall_1': [], 'f1_1': []}
mlp_metrics = {'accuracy': [], 'precision_0': [], 'recall_0': [], 'f1_0': [],
               'precision_1': [], 'recall_1': [], 'f1_1': []}

# --- Cross-Validation Loop ---
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    # Split data into training and validation sets for this fold
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Handle class imbalance using SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    # Feature selection using Logistic Regression with balanced class weights
    sfm = SelectFromModel(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    sfm.fit(X_train_res_scaled, y_train_res)
    X_train_res_selected = sfm.transform(X_train_res_scaled)
    X_val_selected = sfm.transform(X_val_scaled)

    # --- Bagged Logistic Regression ---
    print("Training Bagged Logistic Regression...")
    bagged_lr = BaggingClassifier(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        n_estimators=10, max_samples=0.5, random_state=42)
    bagged_lr.fit(X_train_res_selected, y_train_res)
    lr_probs = bagged_lr.predict_proba(X_val_selected)[:, 1]
    lr_thresh = optimize_threshold(y_val, lr_probs)
    y_pred_lr = (lr_probs >= lr_thresh).astype(int)
    report_lr = classification_report(y_val, y_pred_lr, output_dict=True, zero_division=0)
    extract_metrics(report_lr, lr_metrics)

    # --- Random Forest Classifier ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=1,
        max_features='sqrt', class_weight='balanced_subsample', random_state=42)
    rf.fit(X_train_res_selected, y_train_res)
    rf_probs = rf.predict_proba(X_val_selected)[:, 1]
    rf_thresh = optimize_threshold(y_val, rf_probs)
    y_pred_rf = (rf_probs >= rf_thresh).astype(int)
    report_rf = classification_report(y_val, y_pred_rf, output_dict=True, zero_division=0)
    extract_metrics(report_rf, rf_metrics)

    # --- Multi-layer Perceptron (MLP) Neural Network ---
    print("Training MLP Neural Network...")
    tf.keras.backend.clear_session()
    mlp = Sequential([
        Input(shape=(X_train_res_scaled.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    mlp.fit(X_train_res_scaled, y_train_res,
            validation_split=0.2, epochs=100, batch_size=32,
            verbose=0, callbacks=[early_stop])
    mlp_probs = mlp.predict(X_val_scaled).ravel()
    mlp_thresh = optimize_threshold(y_val, mlp_probs)
    y_pred_mlp = (mlp_probs >= mlp_thresh).astype(int)
    report_mlp = classification_report(y_val, y_pred_mlp, output_dict=True, zero_division=0)
    extract_metrics(report_mlp, mlp_metrics)

# --- Results Presentation Function ---
def print_results(model_name, metrics, y_val, y_pred):
    """
    Print model performance metrics and a confusion matrix heatmap.

    Parameters
    ----------
    model_name : str
        Name of the model.
    metrics : dict
        Dictionary containing performance metrics.
    y_val : pd.Series
        True labels from the last fold.
    y_pred : np.ndarray
        Predicted labels from the last fold.
    """
    mean_acc = np.mean(metrics['accuracy'])
    std_acc = np.std(metrics['accuracy'])
    eger = 1 - mean_acc
    std_eger = std_acc  # Conservative estimate since EGER = 1 - accuracy

    print(f"\n{model_name} Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Estimated Generalisation Error Rate (EGER): {eger:.4f} (+/- {std_eger:.4f})\n")
    print(f"Classification Report (averaged across folds):\n")
    print(f"              precision           recall              f1-score    (± std)")
    print(f"")
    print(f"Class 0       {np.mean(metrics['precision_0']):.4f} (±{np.std(metrics['precision_0']):.4f})"
          f"    {np.mean(metrics['recall_0']):.4f} (±{np.std(metrics['recall_0']):.4f})"
          f"    {np.mean(metrics['f1_0']):.4f} (±{np.std(metrics['f1_0']):.4f})")
    print(f"Class 1       {np.mean(metrics['precision_1']):.4f} (±{np.std(metrics['precision_1']):.4f})"
          f"    {np.mean(metrics['recall_1']):.4f} (±{np.std(metrics['recall_1']):.4f})"
          f"    {np.mean(metrics['f1_1']):.4f} (±{np.std(metrics['f1_1']):.4f})\n")
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix (Last Fold)")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# --- Print Results for Each Model ---
print_results("Bagged Logistic Regression", lr_metrics, y_val, y_pred_lr)
print_results("Random Forest", rf_metrics, y_val, y_pred_rf)
print_results("MLP Neural Network", mlp_metrics, y_val, y_pred_mlp)


