import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

# CONFIG
DATA_FILE = './data/hand_data_clean.csv'  # <--- ENSURE THIS IS THE CLEAN DATA
MODEL_FILE = './model/sign_language_model.p'

if not os.path.exists('./model'):
    os.makedirs('./model')

print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: {DATA_FILE} not found. You must run 'src/fix_data.py' first!")
    exit()

# Sanity Check: Print what classes we found
# We grab the first column as labels
labels = df.iloc[:, 0].unique()
print(f"Found {len(labels)} classes: {labels}")

# Separate Features (Landmarks) and Labels
X = df.iloc[:, 1:].values  # All columns except the first (features)
y = df.iloc[:, 0].values   # The first column (label)

print(f"Data Loaded: {len(df)} samples with {X.shape[1]} features.")

# Split Data
# Stratify ensures we have equal amounts of 'A', 'B', 'C' in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model (Random Forest with optimized parameters)
# These settings help prevent the "Everything is K" issue by reducing overfitting
print("Training Model...")
model = RandomForestClassifier(
    n_estimators=100,      # More trees for stability
    max_depth=10,          # Prevent memorizing exact noise
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

# Check for overfitting warning
if train_accuracy - test_accuracy > 10:
    print("⚠️ Warning: Model might be overfitting (train accuracy >> test accuracy)")

# Save Model
# We save it as a dictionary so inference.py can load it easily
model_dict = {
    'model': model,
    'classes': labels  # Saving the class list just in case we need it later
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_dict, f)

print(f"\n✅ SUCCESS: Model saved to {MODEL_FILE}")