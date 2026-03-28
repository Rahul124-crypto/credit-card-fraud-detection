"""
Train a machine learning model for credit card fraud detection.
This script creates a synthetic dataset, preprocesses it, trains a model,
and saves it for later use in the Flask application.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def create_synthetic_data(n_samples=10000):
    """
    Create synthetic credit card transaction data.
    
    Features:
    - Time: Hours since first transaction (0-48 hours)
    - Amount: Transaction amount (0-25000)
    - V1-V6: Anonymized features (simulated PCA components)
    - Fraud: Target variable (0=legitimate, 1=fraudulent)
    """
    print("Creating synthetic dataset...")
    
    np.random.seed(42)
    
    # Create legitimate transactions (majority class - about 99.5%)
    n_legitimate = int(n_samples * 0.995)
    legitimate_time = np.random.uniform(0, 48, n_legitimate)
    legitimate_amount = np.random.exponential(50, n_legitimate)
    legitimate_v1 = np.random.normal(0, 2, n_legitimate)
    legitimate_v2 = np.random.normal(0, 1.5, n_legitimate)
    legitimate_v3 = np.random.normal(0, 1, n_legitimate)
    legitimate_v4 = np.random.normal(0, 1.5, n_legitimate)
    legitimate_v5 = np.random.normal(0, 1, n_legitimate)
    legitimate_v6 = np.random.normal(0, 1, n_legitimate)
    legitimate_fraud = np.zeros(n_legitimate)
    
    # Create fraudulent transactions (minority class - about 0.5%)
    n_fraud = int(n_samples * 0.005)
    fraud_time = np.random.uniform(2, 4, n_fraud)  # Unusual time patterns
    fraud_amount = np.random.exponential(500, n_fraud)  # Larger amounts
    fraud_v1 = np.random.normal(5, 2, n_fraud)  # Different distributions
    fraud_v2 = np.random.normal(-5, 1.5, n_fraud)
    fraud_v3 = np.random.normal(3, 1, n_fraud)
    fraud_v4 = np.random.normal(-4, 1.5, n_fraud)
    fraud_v5 = np.random.normal(2, 1, n_fraud)
    fraud_v6 = np.random.normal(-3, 1, n_fraud)
    fraud_fraud = np.ones(n_fraud)
    
    # Combine datasets
    X = np.vstack([
        np.column_stack([legitimate_time, legitimate_amount, 
                        legitimate_v1, legitimate_v2, legitimate_v3, 
                        legitimate_v4, legitimate_v5, legitimate_v6]),
        np.column_stack([fraud_time, fraud_amount, 
                        fraud_v1, fraud_v2, fraud_v3, 
                        fraud_v4, fraud_v5, fraud_v6])
    ])
    
    y = np.hstack([legitimate_fraud, fraud_fraud])
    
    # Shuffle data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def preprocess_data(X, y, fit_scaler=True, scaler=None):
    """
    Preprocess data: handle imbalance using SMOTE and scale features.
    """
    print("Preprocessing data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance (only on training data)
    print(f"Before SMOTE - Train: {np.bincount(y_train.astype(int))}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train: {np.bincount(y_train_smote.astype(int))}")
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_smote, y_test, scaler

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train the fraud detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: 'logistic' or 'random_forest'
    
    Returns:
        Trained model
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the model and display metrics.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # ROC-AUC Score
    y_test_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Legitimate', 'Fraudulent']))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix image
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/confusion_matrix.png', dpi=100, bbox_inches='tight')
    print("\nConfusion matrix image saved to static/confusion_matrix.png")
    plt.close()
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def save_model(model, scaler, model_filename='model/fraud_model.pkl', 
               scaler_filename='model/scaler.pkl'):
    """
    Save the trained model and scaler to disk.
    """
    os.makedirs('model', exist_ok=True)
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("\n" + "="*60)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_data(n_samples=10000)
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud ratio: {np.mean(y):.4f}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model (using Logistic Regression)
    # You can change this to 'random_forest' for better performance
    model = train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("You can now run the Flask app with: python app.py")

if __name__ == '__main__':
    main()
