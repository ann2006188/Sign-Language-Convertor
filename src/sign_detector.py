import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

class SignDetector:
    def __init__(self, model_path='./model/sign_language_model.p'):
        """
        Initializes the model and MediaPipe.
        """
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
            
        model_dict = pickle.load(open(model_path, 'rb'))
        self.model = model_dict['model']

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        )

    def predict(self, frame):
        """
        Input: A single video frame (image).
        Output: The predicted character (str) or None if no hand is found.
        """
        # 1. Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # 2. If hand found, process it
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Optional: Draw landmarks on the frame so the user sees the skeleton
                # (Your teammate can decide to show this frame or the original)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 3. Extract Features (x, y)
                data = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data.append(lm.x - min(x_))
                    data.append(lm.y - min(y_))

                # 4. Predict
                prediction = self.model.predict([np.asarray(data)])
                return prediction[0] # Return the letter (e.g., "A", "Hello")
        
        return None # No hand detected