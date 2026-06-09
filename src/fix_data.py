import pandas as pd
import numpy as np
import os

INPUT_FILE = './data/hand_data.csv' 
OUTPUT_FILE = './data/hand_data_clean.csv'

print("Reading raw data...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("Error: No data found! Run collect_custom.py first.")
    exit()

print(f"Processing {len(df)} samples...")

data_clean = []

for index, row in df.iterrows():
    label = row.iloc[0]
    # Get all landmarks
    landmarks = row.iloc[1:].values.astype(float).reshape(-1, 2)
    
    # === WRIST-ANCHOR NORMALIZATION ===
    # 1. Wrist is always at index 0
    wrist_x, wrist_y = landmarks[0][0], landmarks[0][1]
    
    # 2. Shift everything so Wrist is at (0,0)
    shifted_landmarks = []
    for x, y in landmarks:
        shifted_landmarks.append([x - wrist_x, y - wrist_y])
    
    shifted_landmarks = np.array(shifted_landmarks)
    
    # 3. Scale Normalization (Divide by max value to make size consistent)
    # This fixes the "Hand too close/too far" issue
    max_value = np.max(np.abs(shifted_landmarks))
    if max_value == 0: max_value = 1 # Prevent divide by zero
    
    normalized_landmarks = (shifted_landmarks / max_value).flatten().tolist()
    # ==================================
        
    data_clean.append([label] + normalized_landmarks)

# Save
header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}'])

df_clean = pd.DataFrame(data_clean, columns=header)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"SUCCESS: converted to WRIST-RELATIVE coordinates in {OUTPUT_FILE}")
print("NOW: Run src/train_model.py again.")