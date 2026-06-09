import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Fixed Connections from MediaPipe
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                   (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

class SignDetector:
    def __init__(self, model_path='./model/sign_dl_model.pth', classes_path='./model/classes.p', backbone_path='./model/best_backbone.p'):
        """
        Initializes the Deep Learning model and MediaPipe.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
            
        try:
            with open(classes_path, 'rb') as f:
                self.classes = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading classes: {e}")
            
        try:
            with open(backbone_path, 'rb') as f:
                self.backbone_name = pickle.load(f)
        except Exception as e:
            # Fallback
            self.backbone_name = "resnet18"
            print(f"Warning: backbone info not found, defaulting to {self.backbone_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._get_model(len(self.classes), self.backbone_name)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Error loading PyTorch model: {e}")
            
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Initialize MediaPipe Tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path='./model/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.detector_mp = vision.HandLandmarker.create_from_options(options)

    def _get_model(self, num_classes, backbone_name):
        if backbone_name == 'resnet18':
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def _render_skeleton_image(self, normalized_landmarks, size=224):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        points = []
        for i in range(21):
            x_val = normalized_landmarks[i * 2]
            y_val = normalized_landmarks[i * 2 + 1]
            
            x = int(((x_val + 1) / 2.0) * size * 0.8 + size * 0.1)
            y = int(((y_val + 1) / 2.0) * size * 0.8 + size * 0.1)
            
            x = max(0, min(size - 1, x))
            y = max(0, min(size - 1, y))
            
            points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(img, points[p1], points[p2], (255, 255, 255), 2)
            
        return img

    def predict(self, frame):
        import time
        start_time = time.time()
        """
        Input: A single video frame (image).
        Output: The predicted character (str) or None if no hand is found.
        """
        # 1. Convert frame to mp Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.detector_mp.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Extract coordinates
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append([lm.x, lm.y])
                
                # Draw on original frame Manually
                h, w, _ = frame.shape
                for p1, p2 in HAND_CONNECTIONS:
                    if p1 < len(landmarks) and p2 < len(landmarks):
                        x1, y1 = int(landmarks[p1][0]*w), int(landmarks[p1][1]*h)
                        x2, y2 = int(landmarks[p2][0]*w), int(landmarks[p2][1]*h)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.circle(frame, (x1, y1), 3, (0, 0, 255), -1)
                        if p2 == len(landmarks) - 1:
                            cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)
                
                landmarks = np.array(landmarks)
                wrist = landmarks[0]
                shifted_landmarks = landmarks - wrist
                
                max_value = np.max(np.abs(shifted_landmarks))
                if max_value == 0: 
                    max_value = 1
                
                normalized_landmarks = (shifted_landmarks / max_value).flatten().tolist()
                
                # Render the skeletal image specifically for the model
                skeleton_img = self._render_skeleton_image(normalized_landmarks)
                
                # Prepare tensor
                tensor_img = self.transform(skeleton_img).unsqueeze(0).to(self.device)
                
                # Predict
                try:
                    with torch.no_grad():
                        outputs = self.model(tensor_img)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        class_idx = predicted.item()
                        conf_score = confidence.item()
                        
                        end_time = time.time()
                        process_time = (end_time - start_time) * 1000 # in ms
                        print(f"Backbone: {self.backbone_name} | Pred: {self.classes[class_idx]} ({conf_score:.2%}) | Time: {process_time:.1f}ms")
                        
                        return self.classes[class_idx]
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    return "?"
        
        return None