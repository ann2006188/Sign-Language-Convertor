import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Load the trained model
try:
    with open('./model/sign_language_model.p', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERROR: Model not found. Did you run 'src/train_model.py'?")
    exit()

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 3. Start Webcam
cap = cv2.VideoCapture(0)

print("Starting Inference... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip frame for natural interaction (mirror effect)
    frame = cv2.flip(frame, 1)
    
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect the data (Must match how we trained it!)
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])

            # Make Prediction
            prediction = model.predict([np.asarray(data)])
            predicted_character = prediction[0]

            # Display the result
            cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, predicted_character, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()