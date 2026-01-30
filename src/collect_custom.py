import cv2
import mediapipe as mp
import csv
import os
import time

# === CONFIG ===
OUTPUT_FILE = './data/hand_data.csv'
SAMPLES_PER_LETTER = 200
RAW_DATA_PATH = './raw_data' # Path to your A-Z folders
# ==============

# 1. SETUP MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 2. FILE SETUP (Safe Append Mode)
# We only create the header if the file does NOT exist.
if not os.path.exists(OUTPUT_FILE):
    print(f"Creating NEW data file: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)
else:
    print(f"Found EXISTING file: {OUTPUT_FILE}. New data will be APPENDED.")

cap = cv2.VideoCapture(0)

# 3. DEFINE ALPHABET & SELECTION MENU
FULL_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
                 'space', 'del', 'nothing', 'Hello', 'Thanks', 'Yes']

print("\n=== DATA COLLECTION MODE ===")
print("1. Record FULL Alphabet (A-Z)")
print("2. Record SINGLE Specific Letter (Add to existing)")
choice = input("Enter 1 or 2: ")

if choice == '1':
    TARGET_LIST = FULL_ALPHABET
else:
    target = input("Enter the label to record (e.g., K, Hello): ")
    TARGET_LIST = [target]

def get_overlay_image(letter):
    """Tries to load a reference image for the letter from raw_data"""
    folder_path = os.path.join(RAW_DATA_PATH, letter)
    if os.path.exists(folder_path):
        images = os.listdir(folder_path)
        if images:
            img_path = os.path.join(folder_path, images[0])
            img = cv2.imread(img_path)
            if img is not None:
                return cv2.resize(img, (150, 150))
    return None

print(f"\nStarting collection for: {TARGET_LIST}")

for letter in TARGET_LIST:
    ref_img = get_overlay_image(letter)
    
    # --- PREPARATION PHASE (Press Enter) ---
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # Overlay Reference Image
        if ref_img is not None:
            h, w, _ = ref_img.shape
            frame[10:10+h, -160:-160+w] = ref_img
            cv2.putText(frame, "MIMIC:", (frame.shape[1]-160, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Instructions
        cv2.putText(frame, f"TARGET: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "Press 'ENTER' to start", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Mode: APPEND", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Collector', frame)
        key = cv2.waitKey(1)
        if key == 13: # Enter key (Ascii 13)
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # --- COUNTDOWN PHASE (3 SECONDS) ---
    start_time = time.time()
    while int(time.time() - start_time) < 3: 
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        if ref_img is not None:
            h, w, _ = ref_img.shape
            frame[10:10+h, -160:-160+w] = ref_img
            
        countdown = 3 - int(time.time() - start_time)
        
        cv2.putText(frame, str(countdown), (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 6)
        cv2.putText(frame, f"GET READY: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Collector', frame)
        cv2.waitKey(1)

    # --- RECORDING PHASE ---
    print(f"Collecting {letter}...")
    count = 0
    while count < SAMPLES_PER_LETTER:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                
                # 'a' mode ensures we APPEND, not overwrite
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([letter] + data)
                
                count += 1
                
        # Progress Bar
        bar_width = int(500 * (count / SAMPLES_PER_LETTER))
        cv2.rectangle(frame, (0, 400), (bar_width, 450), (0, 255, 0), -1)
        cv2.putText(frame, f"{letter}: {count}/{SAMPLES_PER_LETTER}", (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Collector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

print("\nDone! REMINDER: You must run 'src/fix_data.py' now to normalize this new data!")
cap.release()
cv2.destroyAllWindows()