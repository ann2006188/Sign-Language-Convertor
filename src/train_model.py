import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

# CONFIG
DATA_FILE = './data/hand_data.csv'
MODEL_FILE = './model/sign_language_model.p'

if not os.path.exists('./model'):
    os.makedirs('./model')

print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Separate Features (Landmarks) and Labels
X = df.iloc[:, 1:].values  # All columns except the first
y = df.iloc[:, 0].values   # The first column (label)

print(f"Data Loaded: {len(df)} samples with {X.shape[1]} features.")
print(f"Unique labels: {np.unique(y)}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model (Random Forest with optimized parameters)
print("Training Model...")
model = RandomForestClassifier(
    n_estimators=100,      # More trees for better accuracy
    max_depth=10,          # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

# Check for overfitting
if train_accuracy - test_accuracy > 10:
    print("⚠️ Warning: Model might be overfitting (train accuracy >> test accuracy)")

# Get feature importances (optional - helps debug)
feature_importance = model.feature_importances_
print(f"\nTop 5 most important features: {np.argsort(feature_importance)[-5:][::-1]}")

# Create labels dictionary (maps index to label)
unique_labels = np.unique(y)
labels_dict = {i: label for i, label in enumerate(unique_labels)}

# Save Model as Dictionary (FIXED!)
model_dict = {
    'model': model,
    'labels': labels_dict,
    'accuracy': test_accuracy,
    'n_features': X.shape[1],
    'n_samples': len(df)
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_dict, f)

print(f"\n✅ SUCCESS: Model saved to {MODEL_FILE}")
print(f"   - Model type: {type(model).__name__}")
print(f"   - Test accuracy: {test_accuracy:.2f}%")
print(f"   - Number of classes: {len(unique_labels)}")
print(f"   - Classes: {unique_labels}")