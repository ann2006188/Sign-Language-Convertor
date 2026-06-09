import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import time

# === CONFIG ===
OUTPUT_FILE      = './data/hand_data.csv'
SAMPLES_PER_LETTER = 100
RAW_DATA_PATH    = './raw_data'
TASK_MODEL_PATH  = './model/hand_landmarker.task'
# ==============

# 1. SETUP MEDIAPIPE TASKS API
base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options      = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

# Hand skeleton connections for manual drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def detect_landmarks(frame_bgr):
    """Run hand landmarker on a BGR frame; returns list of 21 (x,y) pairs or None."""
    rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result    = hand_landmarker.detect(mp_image)
    if result.hand_landmarks:
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
    return None

def draw_landmarks(frame, landmarks):
    """Draw landmarks and connections onto a BGR frame in-place."""
    h, w = frame.shape[:2]
    pts  = [(int(x * w), int(y * h)) for x, y in landmarks]
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, pts[p1], pts[p2], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)

# 2. FILE SETUP (Safe Append Mode)
if not os.path.exists(OUTPUT_FILE):
    print(f"Creating NEW data file: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        csv.writer(f).writerow(header)
else:
    print(f"Found EXISTING file: {OUTPUT_FILE}. New data will be APPENDED.")

cap = cv2.VideoCapture(0)

# 3. DEFINE ALPHABET & SELECTION MENU
FULL_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                 'space', 'del', 'nothing', 'Hello', 'Thanks', 'Yes']

print("\n=== DATA COLLECTION MODE ===")
print("1. Record FULL Alphabet (A-Z + extras)")
print("2. Record SINGLE Specific Letter (Add to existing)")
choice = input("Enter 1 or 2: ")

if choice == '1':
    TARGET_LIST = FULL_ALPHABET
else:
    target      = input("Enter the label to record (e.g., K, Hello): ")
    TARGET_LIST = [target]

def get_overlay_image(letter):
    """Tries to load a reference image for the letter from raw_data."""
    folder_path = os.path.join(RAW_DATA_PATH, letter)
    if os.path.exists(folder_path):
        images = os.listdir(folder_path)
        if images:
            img = cv2.imread(os.path.join(folder_path, images[0]))
            if img is not None:
                return cv2.resize(img, (150, 150))
    return None

print(f"\nStarting collection for: {TARGET_LIST}")

for letter in TARGET_LIST:
    ref_img = get_overlay_image(letter)

    # --- PREPARATION PHASE (Press Enter) ---
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        if ref_img is not None:
            h, w, _ = ref_img.shape
            frame[10:10+h, -160:-160+w] = ref_img
            cv2.putText(frame, "MIMIC:", (frame.shape[1]-160, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"TARGET: {letter}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "Press 'ENTER' to start", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Collector', frame)
        key = cv2.waitKey(1)
        if key == 13:   # Enter
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            hand_landmarker.close()
            exit()

    # --- COUNTDOWN PHASE (3 seconds) ---
    start_time = time.time()
    while int(time.time() - start_time) < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        if ref_img is not None:
            h, w, _ = ref_img.shape
            frame[10:10+h, -160:-160+w] = ref_img
        countdown = 3 - int(time.time() - start_time)
        cv2.putText(frame, str(countdown), (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 6)
        cv2.putText(frame, f"GET READY: {letter}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collector', frame)
        cv2.waitKey(1)

    # --- RECORDING PHASE ---
    print(f"Collecting {letter}...")
    count = 0
    while count < SAMPLES_PER_LETTER:
        ret, frame = cap.read()
        if not ret:
            continue
        frame      = cv2.flip(frame, 1)
        landmarks  = detect_landmarks(frame)

        if landmarks:
            clean_frame = frame.copy()
            draw_landmarks(frame, landmarks)
            data = []
            for x, y in landmarks:
                data.extend([x, y])
            with open(OUTPUT_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([letter] + data)
            count += 1
            
            save_dir = os.path.join(RAW_DATA_PATH, letter)
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f"{letter}_{count:03d}.jpg")
            cv2.imwrite(img_path, clean_frame)

        # Progress bar
        bar_w = int(500 * (count / SAMPLES_PER_LETTER))
        cv2.rectangle(frame, (0, 400), (bar_w, 450), (0, 255, 0), -1)
        cv2.putText(frame, f"{letter}: {count}/{SAMPLES_PER_LETTER}",
                    (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Collector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            hand_landmarker.close()
            exit()

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\nDone! REMINDER: Run 'src/fix_data.py' to normalise the new data.")