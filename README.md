# Sign Language Assistive Translator

An assistive, human-centric AI system that enables real-time communication between sign language users and non-sign language users by translating hand gestures into text and speech.

Built as part of a 24-hour hackathon under the theme **AI for Assistive & Human-Centric Technology**.

---

##  Features
- Real-time sign language gesture recognition
- Sign → Text translation
- Text → Speech output for accessibility
- Lightweight and offline-capable
- Human-centric, low-cost design

---

##  System Overview
The system uses computer vision to detect hand landmarks, converts them into meaningful features, and classifies gestures using a lightweight machine learning model. The predicted output is displayed as text and optionally converted into speech.

---

##  Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** MediaPipe Hands, OpenCV  
- **Machine Learning:** Scikit-Learn (KNN / SVM)  
- **Text-to-Speech:** pyttsx3  
- **Speech-to-Text (optional):** SpeechRecognition  
- **Frontend:** Streamlit  

All components are open-source and require no paid APIs.

---

##  Project Structure
    sign-language-assistive-ai/
    │
    ├── vision/ # Hand detection & landmark extraction
    ├── ml/ # Feature processing & gesture classification
    ├── audio/ # Text-to-speech and speech-to-text modules
    ├── ui/ # Streamlit application
    ├── requirements.txt
    └── README.md
