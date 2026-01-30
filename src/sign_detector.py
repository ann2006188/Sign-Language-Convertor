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
            
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                self.model = model_dict['model']
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

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
                # Optional: Draw landmarks on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # --- NEW WRIST-ANCHOR NORMALIZATION ---
                # Extract raw coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                landmarks = np.array(landmarks)
                
                # 1. Wrist Anchor (Index 0 is always the wrist)
                wrist = landmarks[0]
                shifted_landmarks = landmarks - wrist  # Center everything on wrist
                
                # 2. Scale Normalization (Fixes distance issues)
                max_value = np.max(np.abs(shifted_landmarks))
                if max_value == 0: 
                    max_value = 1
                
                normalized_landmarks = (shifted_landmarks / max_value).flatten().tolist()
                # --------------------------------------

                # 3. Predict
                try:
                    prediction = self.model.predict([np.asarray(normalized_landmarks)])
                    return prediction[0] # Return the letter (e.g., "A", "Hello")
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    return "?"
        
        return None # No hand detected