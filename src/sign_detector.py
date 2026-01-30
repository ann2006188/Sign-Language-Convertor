import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time

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
        
        # Get labels mapping if available (for better debugging)
        self.labels = model_dict.get('labels', None)

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.7
        )
        
        # Add prediction smoothing
        self.prediction_buffer = []
        self.buffer_size = 5  # Number of frames to smooth over
        
        # Add confidence tracking
        self.last_prediction = None
        self.prediction_count = 0
        self.min_stable_frames = 3  # Frames needed before confirming prediction

    def predict(self, frame, return_confidence=False):
        """
        Input: A single video frame (image).
        Output: The predicted character (str) or None if no hand is found.
                If return_confidence=True, returns (prediction, confidence)
        """
        # 1. Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # 2. If hand found, process it
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with better styling
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 3. Extract Features (x, y) - normalized
                data = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Normalize coordinates
                min_x, max_x = min(x_), max(x_)
                min_y, max_y = min(y_), max(y_)
                
                for lm in hand_landmarks.landmark:
                    # Normalize to 0-1 range
                    norm_x = (lm.x - min_x) / (max_x - min_x + 1e-6)
                    norm_y = (lm.y - min_y) / (max_y - min_y + 1e-6)
                    data.append(norm_x)
                    data.append(norm_y)

                # 4. Predict with probability
                prediction = self.model.predict([np.asarray(data)])
                predicted_char = prediction[0]
                
                # Get confidence if model supports predict_proba
                try:
                    proba = self.model.predict_proba([np.asarray(data)])
                    confidence = np.max(proba)
                except:
                    confidence = 1.0  # Default if model doesn't support probabilities
                
                # 5. Smooth predictions using buffer
                self.prediction_buffer.append(predicted_char)
                if len(self.prediction_buffer) > self.buffer_size:
                    self.prediction_buffer.pop(0)
                
                # Get most common prediction in buffer
                if len(self.prediction_buffer) >= 3:
                    from collections import Counter
                    most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                    
                    if return_confidence:
                        return most_common, confidence
                    return most_common
                
                if return_confidence:
                    return predicted_char, confidence
                return predicted_char
        
        # Clear buffer if no hand detected
        self.prediction_buffer = []
        
        if return_confidence:
            return None, 0.0
        return None

    def predict_stable(self, frame):
        """
        Returns prediction only when it's stable across multiple frames.
        Useful for word formation - prevents jittery letter detection.
        """
        prediction, confidence = self.predict(frame, return_confidence=True)
        
        if prediction is None:
            self.last_prediction = None
            self.prediction_count = 0
            return None
        
        # Check if prediction is stable
        if prediction == self.last_prediction:
            self.prediction_count += 1
        else:
            self.last_prediction = prediction
            self.prediction_count = 1
        
        # Only return if stable and confident
        if self.prediction_count >= self.min_stable_frames and confidence > 0.7:
            return prediction
        
        return None

    def get_hand_bbox(self, frame):
        """
        Returns bounding box coordinates of detected hand.
        Useful for UI visualization.
        Returns: (x, y, w, h) or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return None

    def close(self):
        """Clean up resources"""
        self.hands.close()