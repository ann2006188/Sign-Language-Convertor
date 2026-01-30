import os
import cv2
import csv
import mediapipe as mp  # <--- STANDARD IMPORT

# Initialize MediaPipe
mp_hands = mp.solutions.hands  # <--- THIS WILL WORK NOW
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# === CONFIGURATION ===
DATA_DIR = './raw_data'
OUTPUT_FILE = './data/hand_data.csv'
LIMIT_PER_CLASS = 1000 
# =====================

# Create the data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

print(f"Creating {OUTPUT_FILE}...")

# Create the CSV file and write the header
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])
    writer.writerow(header)

# Check if raw_data exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory '{DATA_DIR}' not found.")
    exit()

# Get sorted list of all folders
sorted_labels = sorted([d for d in os.listdir(DATA_DIR) 
                       if os.path.isdir(os.path.join(DATA_DIR, d))])

if not sorted_labels:
    print(f"ERROR: No class folders found in '{DATA_DIR}'")
    exit()

print(f"Found {len(sorted_labels)} classes: {sorted_labels}\n")

total_processed = 0

for label in sorted_labels:
    class_dir = os.path.join(DATA_DIR, label)
    
    if label in ['space', 'del', 'nothing']:
        print(f"--> [SPECIAL CLASS]: {label}")
    else:
        print(f"--> Processing: {label}")
    
    count = 0
    skipped = 0
    
    images = [f for f in os.listdir(class_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for img_name in images[:LIMIT_PER_CLASS]:
        img_path = os.path.join(class_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None: 
            skipped += 1
            continue 

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + data)
                
                count += 1
        else:
            skipped += 1
        
        if count > 0 and count % 200 == 0:
            print(f"    {count} processed...")
    
    total_processed += count
    print(f"    ✓ {count} saved, {skipped} skipped\n")

hands.close()

print(f"\n{'='*50}")
print(f"SUCCESS! Total: {total_processed} images")
print(f"File: {OUTPUT_FILE}")
print(f"{'='*50}")