import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv

# === CONFIGURATION ===
DATA_DIR        = './raw_data'
OUTPUT_FILE     = './data/hand_data.csv'
LIMIT_PER_CLASS = 1000
TASK_MODEL_PATH = './model/hand_landmarker.task'
# =====================

# Initialize MediaPipe Tasks API (IMAGE mode — synchronous, good for static files)
base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options      = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

os.makedirs('./data', exist_ok=True)

print(f"Creating {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])
    csv.writer(f).writerow(header)

if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory '{DATA_DIR}' not found.")
    hand_landmarker.close()
    exit()

sorted_labels = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

if not sorted_labels:
    print(f"ERROR: No class folders found in '{DATA_DIR}'")
    hand_landmarker.close()
    exit()

print(f"Found {len(sorted_labels)} classes: {sorted_labels}\n")

total_processed = 0

for label in sorted_labels:
    tag       = "[SPECIAL]" if label in ('space', 'del', 'nothing') else "Processing"
    print(f"--> {tag}: {label}")
    class_dir = os.path.join(DATA_DIR, label)
    count     = 0
    skipped   = 0

    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for img_name in images[:LIMIT_PER_CLASS]:
        img_path = os.path.join(class_dir, img_name)
        image    = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = hand_landmarker.detect(mp_img)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            data      = []
            for lm in landmarks:
                data.extend([lm.x, lm.y])
            with open(OUTPUT_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([label] + data)
            count += 1
        else:
            skipped += 1

        if count > 0 and count % 200 == 0:
            print(f"    {count} processed...")

    total_processed += count
    print(f"    {count} saved, {skipped} skipped\n")

hand_landmarker.close()

print("=" * 50)
print(f"SUCCESS! Total: {total_processed} images")
print(f"File: {OUTPUT_FILE}")
print("=" * 50)