import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

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

print(f"Data Loaded: {len(df)} samples.")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest is faster/better for mixed data)
print("Training Model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Save
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"SUCCESS: Model saved to {MODEL_FILE}")