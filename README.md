# Sign Language Convertor

A real-time sign language recognition system that translates hand gestures into text using deep learning and computer vision. The application captures webcam input, detects hand landmarks using MediaPipe, and predicts gestures through trained deep learning models, displaying results in an interactive Streamlit interface.

Key Features:
- Real-time hand sign detection and recognition
- Gesture-to-text conversion using webcam input
- Hand landmark extraction with MediaPipe
- Deep learning-based classification using ResNet18 and EfficientNet-B0
- Interactive Streamlit web application
- Performance evaluation using confusion matrices and ROC curves

Technologies Used:
- Python
- PyTorch
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- Scikit-learn

Project Workflow:
1. Collect and preprocess gesture images
2. Extract hand landmarks using MediaPipe
3. Train deep learning models for gesture classification
4. Evaluate model performance
5. Deploy the trained model through a Streamlit application

Future Enhancements:
- Continuous sign language sentence recognition
- Text-to-speech conversion
- Expanded gesture vocabulary
- Transformer-based models
- Web deployment

Developed by Ananya Krishnan
AI & ML Undergraduate interested in Computer Vision, Generative AI, and Software Development.