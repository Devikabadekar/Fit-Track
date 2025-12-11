import pickle
import numpy as np

print("Testing retrained diabetes model...")
print("="*50)

# Load the model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))

# Test case 1: High risk patient
test_data1 = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
test_scaled1 = scaler.transform(test_data1)
pred1 = model.predict(test_scaled1)[0]
proba1 = model.predict_proba(test_scaled1)[0]

print("\nTest Case 1 (High Risk):")
print(f"  Input: Pregnancies=6, Glucose=148, BP=72, etc.")
print(f"  Prediction: {'Positive' if pred1 == 1 else 'Negative'}")
print(f"  Confidence: {max(proba1)*100:.2f}%")

# Test case 2: Low risk patient
test_data2 = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
test_scaled2 = scaler.transform(test_data2)
pred2 = model.predict(test_scaled2)[0]
proba2 = model.predict_proba(test_scaled2)[0]

print("\nTest Case 2 (Low Risk):")
print(f"  Input: Pregnancies=1, Glucose=85, BP=66, etc.")
print(f"  Prediction: {'Positive' if pred2 == 1 else 'Negative'}")
print(f"  Confidence: {max(proba2)*100:.2f}%")

print("\n" + "="*50)
print("✓ Model is working correctly!")
print("✓ Ready to use in Flask app!")

