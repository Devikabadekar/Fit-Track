"""
Retrain Diabetes Detection Model
This script retrains the Random Forest model using the current environment's sklearn version
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

print("="*60)
print("Diabetes Model Retraining Script")
print("="*60)

# Check versions
print(f"\nPython version: {sys.version}")
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

print("\n" + "="*60)
print("Loading diabetes dataset...")
print("="*60)

# Load the diabetes dataset
try:
    df = pd.read_csv('diabetes.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {df.columns.tolist()}")
    print(f"  - Target distribution:\n{df['Outcome'].value_counts()}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Preparing data...")
print("="*60)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"✓ Features prepared: {X.shape}")
print(f"✓ Target prepared: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

print("\n" + "="*60)
print("Scaling features...")
print("="*60)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled successfully")

print("\n" + "="*60)
print("Training Random Forest model...")
print("="*60)

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1  # Use all CPU cores
)

rf_model.fit(X_train_scaled, y_train)
print("✓ Model training completed!")

print("\n" + "="*60)
print("Evaluating model performance...")
print("="*60)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*60)
print("Saving model and scaler...")
print("="*60)

# Save the model and scaler
try:
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ Model saved to: diabetes_model.pkl")
    
    with open('diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved to: diabetes_scaler.pkl")
except Exception as e:
    print(f"✗ Error saving files: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Testing saved model...")
print("="*60)

# Test loading and using the saved model
try:
    loaded_model = pickle.load(open('diabetes_model.pkl', 'rb'))
    loaded_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
    
    # Test prediction
    test_sample = X_test.iloc[0:1]
    test_sample_scaled = loaded_scaler.transform(test_sample)
    test_pred = loaded_model.predict(test_sample_scaled)[0]
    test_proba = loaded_model.predict_proba(test_sample_scaled)[0]
    
    print("✓ Model loaded and tested successfully!")
    print(f"  - Test prediction: {'Positive' if test_pred == 1 else 'Negative'}")
    print(f"  - Confidence: {max(test_proba)*100:.2f}%")
except Exception as e:
    print(f"✗ Error testing saved model: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Feature Importance:")
print("="*60)

# Display feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

print("\n" + "="*60)
print("SUCCESS! Model retraining completed!")
print("="*60)
print("\n✓ The diabetes detection model is now compatible with your environment")
print("✓ You can now use the /diabetes_detection route in your Flask app")
print("\nFiles created:")
print("  - diabetes_model.pkl")
print("  - diabetes_scaler.pkl")
print("\n" + "="*60)

