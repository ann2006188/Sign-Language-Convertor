import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# CONFIG
DATA_FILE = './data/hand_data_clean.csv'
MODEL_FILE = './model/sign_language_model.p'

if not os.path.exists('./model'):
    os.makedirs('./model')

print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: {DATA_FILE} not found. Did you run collect_custom.py?")
    exit()

# Sanity Check: Print what classes we found
labels = df.iloc[:, 0].unique()
print(f"Found {len(labels)} classes: {labels}")

# Separate Features (Landmarks) and Labels
X = df.iloc[:, 1:].values  # All columns except the first
y = df.iloc[:, 0].values   # The first column (label)

print(f"Data Loaded: {len(df)} samples.")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
print("Training Random Forest...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")

# --- CRITICAL FIX BELOW ---
# We save as a dictionary to match inference.py structure
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"SUCCESS: Model saved to {MODEL_FILE}")