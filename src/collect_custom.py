import cv2
import mediapipe as mp
import csv
import os
import time

# === CONFIG ===
OUTPUT_FILE = './data/hand_data.csv' 
SAMPLES_TO_COLLECT = 200
# ==============

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Ask user what they are recording
label_name = input("Enter the label name (e.g., Hello): ")

print(f"\nGET READY! Recording '{label_name}' in 5 seconds...")

# 5-Second Countdown
start_time = time.time()
while int(time.time() - start_time) < 5:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Show countdown on screen
    countdown = 5 - int(time.time() - start_time)
    cv2.putText(frame, str(countdown), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
    cv2.imshow('Collector', frame)
    cv2.waitKey(1)

print(f"GO! Collecting {label_name}...")

count = 0
while count < SAMPLES_TO_COLLECT:
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
            
            # Append to existing CSV
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label_name] + data)
            
            count += 1
            # Simple progress bar
            cv2.rectangle(frame, (0, 0), (int(300 * (count/SAMPLES_TO_COLLECT)), 50), (0, 255, 0), -1)
            cv2.putText(frame, f"{count}/{SAMPLES_TO_COLLECT}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            print(f"\rCaptured {count}/{SAMPLES_TO_COLLECT}", end="")
            
    else:
        cv2.putText(frame, "HAND NOT FOUND!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nDone! Added {label_name} to dataset.")
cap.release()
cv2.destroyAllWindows()