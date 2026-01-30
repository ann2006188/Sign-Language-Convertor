import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import Counter

# 1. Load the trained model
try:
    with open('./model/sign_language_model.p', 'rb') as f:
        model_data = pickle.load(f)
        
        # Handle both dictionary and direct model formats
        if isinstance(model_data, dict):
            model = model_data['model']
            labels_dict = model_data.get('labels', {})
        else:
            model = model_data
            labels_dict = {}
            
except FileNotFoundError:
    print("ERROR: Model not found. Run 'src/train_model.py' first.")
    exit()
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit()

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.4,  # Reduced to 40%
    min_tracking_confidence=0.4
)

# 3. Prediction smoothing
prediction_buffer = []
BUFFER_SIZE = 5
CONFIDENCE_THRESHOLD = 0.4  # Reduced to 40%

# 4. Start Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Sign Language Detection Started...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_prediction = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks (minimal)
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract normalized features
            data = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            
            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - min_x) / (max_x - min_x + 1e-6)
                norm_y = (lm.y - min_y) / (max_y - min_y + 1e-6)
                data.extend([norm_x, norm_y])

            # Predict
            try:
                prediction = model.predict([np.asarray(data)])
                predicted_character = prediction[0]
                
                # Get confidence
                try:
                    proba = model.predict_proba([np.asarray(data)])
                    confidence = np.max(proba)
                except:
                    confidence = 1.0
                
                # Smooth predictions
                prediction_buffer.append(predicted_character)
                if len(prediction_buffer) > BUFFER_SIZE:
                    prediction_buffer.pop(0)
                
                if len(prediction_buffer) >= 3:
                    most_common = Counter(prediction_buffer).most_common(1)[0][0]
                    current_prediction = most_common
                else:
                    current_prediction = predicted_character
                
                # Print prediction to console (backend output)
                if confidence > CONFIDENCE_THRESHOLD:
                    print(f"Detected: {current_prediction} (Confidence: {confidence:.2f})")
                    
            except Exception as e:
                print(f"Prediction error: {e}")
                current_prediction = None
    else:
        prediction_buffer = []

    # Simple display
    cv2.imshow('Sign Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Detection stopped.")