import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load Model
try:
    with open('./model/sign_language_model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except FileNotFoundError:
    print("Model not found!")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === MATCHING NORMALIZATION LOGIC ===
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])
            
            landmarks = np.array(landmarks)
            
            # 1. Wrist Anchor (Index 0)
            wrist = landmarks[0]
            shifted = landmarks - wrist
            
            # 2. Scale Normalization
            max_value = np.max(np.abs(shifted))
            if max_value == 0: max_value = 1
            
            normalized = (shifted / max_value).flatten().tolist()
            # ====================================

            try:
                prediction = model.predict([np.asarray(normalized)])
                predicted_char = prediction[0]
                confidence = np.max(model.predict_proba([np.asarray(normalized)]))
                
                # Only show if confident (optional filter)
                if confidence > 0.5:
                    disp_text = f"{predicted_char} ({int(confidence*100)}%)"
                else:
                    disp_text = "..."

            except:
                disp_text = "?"

            cv2.rectangle(frame, (0, 0), (350, 70), (0, 0, 0), -1)
            cv2.putText(frame, disp_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('Final Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()