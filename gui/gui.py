import streamlit as st
from gtts import gTTS
import tempfile
import cv2
import numpy as np
import sys
import os
import time
from collections import Counter
import pickle
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Page config
st.set_page_config(
    layout="wide",
    page_title="ASL Converter",
    page_icon="🪧"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #00ff00;
    }
    .word-display {
        font-size:35px !important;
        font-weight: bold;
        color: #ffff00;
        background-color: #333;
        padding: 10px;
        border-radius: 5px;
    }
    .sentence-display {
        font-size:25px !important;
        color: #00ffff;
        background-color: #222;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤟 ASL Converter")
st.caption("Real-time Sign Language to Speech Conversion")

# Sidebar controls
mode = st.sidebar.radio(
    "Select Mode",
    ["👋 Sign → Text → Speech", "🎤 Speech → Text → Sign"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")

# Text-to-speech function
def speak(text):
    """Generate and play speech from text"""
    try:
        tts = gTTS(text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            st.audio(f.name, format='audio/mp3', autoplay=True)
        time.sleep(0.5)
        try:
            os.unlink(f.name)
        except:
            pass
    except Exception as e:
        st.error(f"Speech error: {e}")

# Enhanced Sign Detector Class
class EnhancedSignDetector:
    def __init__(self, model_path, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_dict = pickle.load(open(model_path, 'rb'))
        
        if isinstance(model_dict, dict):
            self.model = model_dict['model']
            self.labels = model_dict.get('labels', {})
        else:
            self.model = model_dict
            self.labels = {}
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Prediction smoothing
        self.prediction_buffer = []
        self.buffer_size = 5
    
    def update_confidence(self, min_detection_confidence, min_tracking_confidence):
        """Update confidence thresholds and recreate hands object"""
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
    def predict(self, frame, return_confidence=False, draw_landmarks=True):
        """Predict sign with confidence and optional landmark drawing"""
        if frame is None or frame.size == 0:
            if return_confidence:
                return None, 0.0
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    if draw_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract and normalize features
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
                    prediction = self.model.predict([np.asarray(data)])
                    predicted_char = prediction[0]
                    
                    # Get confidence
                    try:
                        proba = self.model.predict_proba([np.asarray(data)])
                        confidence = np.max(proba)
                    except:
                        confidence = 1.0
                    
                    # Smooth predictions
                    self.prediction_buffer.append(predicted_char)
                    if len(self.prediction_buffer) > self.buffer_size:
                        self.prediction_buffer.pop(0)
                    
                    if len(self.prediction_buffer) >= 3:
                        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                        if return_confidence:
                            return most_common, confidence
                        return most_common
                    
                    if return_confidence:
                        return predicted_char, confidence
                    return predicted_char
            
            self.prediction_buffer = []
        except Exception as e:
            print(f"Prediction error: {e}")
        
        if return_confidence:
            return None, 0.0
        return None
    
    def get_hand_bbox(self, frame):
        """Get bounding box of detected hand"""
        if frame is None or frame.size == 0:
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                hand_landmarks = results.multi_hand_landmarks[0]
                
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                return (x_min, y_min, x_max - x_min, y_max - y_min)
        except:
            pass
        
        return None
    
    def close(self):
        self.hands.close()

# Load detector
@st.cache_resource
def load_detector():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../model/sign_language_model.p')
        return EnhancedSignDetector(model_path=model_path)
    except FileNotFoundError:
        st.error("❌ Model not found. Please train the model first by running `train_model.py`")
        st.stop()

detector = load_detector()

# ===== SIGN TO SPEECH MODE =====
if mode == "👋 Sign → Text → Speech":
    st.markdown("---")
    
    # Settings - placed BEFORE camera starts
    st.sidebar.markdown("### Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.3, 1.0, 0.4, 0.05,
        help="Minimum confidence for predictions",
        key="confidence_slider"
    )
    
    char_hold_time = st.sidebar.slider(
        "Character Hold Time (sec)", 
        0.5, 3.0, 1.5, 0.1,
        help="How long to hold a sign before adding to word",
        key="hold_time_slider"
    )
    
    show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True, key="landmarks_checkbox")
    
    # Update detector confidence
    detector.update_confidence(confidence_threshold, confidence_threshold)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        stframe = st.empty()
    
    with col2:
        st.subheader("📝 Recognition Results")
        current_sign_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.markdown("### Current Word")
        word_placeholder = st.empty()
        
        st.markdown("### Completed Sentence")
        sentence_placeholder = st.empty()
        
        # Audio placeholder
        audio_placeholder = st.empty()
        
        # Control buttons
        st.markdown("---")
        st.markdown("#### 🎯 Word Controls")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            speak_word_btn = st.button("🔊 Speak Word", help="Speak current word and add to sentence")
        with col_btn2:
            delete_letter_btn = st.button("⌫ Delete Letter", help="Remove last letter from word")
        
        col_btn3, col_btn4 = st.columns(2)
        with col_btn3:
            clear_word_btn = st.button("🗑️ Clear Word", help="Clear current word without speaking")
        with col_btn4:
            add_space_btn = st.button("␣ Add Space", help="Add space to sentence")
        
        st.markdown("#### 📝 Sentence Controls")
        col_btn5, col_btn6 = st.columns(2)
        with col_btn5:
            speak_sentence_btn = st.button("🔊 Speak Sentence", help="Speak entire sentence")
        with col_btn6:
            reset_all_btn = st.button("🔄 Reset All", help="Clear everything")
    
    # Initialize session state
    if 'current_word' not in st.session_state:
        st.session_state.current_word = []
    if 'sentence' not in st.session_state:
        st.session_state.sentence = []
    if 'last_stable_char' not in st.session_state:
        st.session_state.last_stable_char = None
    if 'last_char_time' not in st.session_state:
        st.session_state.last_char_time = time.time()
    if 'is_speaking' not in st.session_state:
        st.session_state.is_speaking = False
    
    # Instructions
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        **Building Words:**
        1. Hold a sign steady for ~1.5 seconds to add it to your current word
        2. Watch the progress bar fill up as you hold
        3. Letters accumulate in the "Current Word" box
        
        **Speaking:**
        - Click "🔊 Speak Word" to hear the current word (adds to sentence)
        - Click "🔊 Speak Sentence" to hear all words together
        
        **Editing:**
        - "⌫ Delete Letter" removes the last letter
        - "🗑️ Clear Word" removes the whole word without speaking
        - "␣ Add Space" adds a space to separate words in sentence
        - "🔄 Reset All" clears everything
        """)
    
    # Start/Stop camera
    run = st.checkbox("🎥 Start Camera", value=False, key="camera_toggle")
    
    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Status indicator
        status_text = st.empty()
        status_text.success("✅ Camera active - Hold signs steady to add letters")
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to access camera.")
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            try:
                frame = cv2.flip(frame, 1)
                H, W, _ = frame.shape
                
                # Ensure valid dimensions
                if H <= 0 or W <= 0:
                    continue
                
                # Get prediction (only if not speaking)
                if not st.session_state.is_speaking:
                    prediction, confidence = detector.predict(
                        frame, 
                        return_confidence=True, 
                        draw_landmarks=show_landmarks
                    )
                    
                    # Update display with prediction
                    if prediction and confidence > confidence_threshold:
                        # Draw bounding box
                        bbox = detector.get_hand_bbox(frame)
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display current prediction
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (0, 0), (W, 100), (0, 0, 0), -1)
                        cv2.putText(frame, f"Sign: {prediction}", (20, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
                        cv2.putText(frame, f"{confidence:.2f}", (W - 150, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        
                        # Word formation logic
                        if prediction == st.session_state.last_stable_char:
                            if time.time() - st.session_state.last_char_time > char_hold_time:
                                st.session_state.current_word.append(prediction)
                                st.session_state.last_char_time = time.time()
                                st.session_state.last_stable_char = None
                                status_text.success(f"✅ Added '{prediction}' to word!")
                        else:
                            st.session_state.last_stable_char = prediction
                            st.session_state.last_char_time = time.time()
                        
                        # Draw progress bar
                        if st.session_state.last_stable_char:
                            hold_progress = min(1.0, (time.time() - st.session_state.last_char_time) / char_hold_time)
                            bar_width = int(300 * hold_progress)
                            cv2.rectangle(frame, (20, H - 50), (20 + bar_width, H - 30), (0, 255, 0), -1)
                            cv2.rectangle(frame, (20, H - 50), (320, H - 30), (255, 255, 255), 2)
                            cv2.putText(frame, "Hold...", (340, H - 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Update sidebar display
                        current_sign_placeholder.markdown(f'<p class="big-font">{prediction}</p>', unsafe_allow_html=True)
                        confidence_placeholder.progress(float(confidence))
                        
                    else:
                        # No hand detected
                        cv2.rectangle(frame, (0, 0), (W, 100), (0, 0, 0), -1)
                        cv2.putText(frame, "No hand detected", (20, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3, cv2.LINE_AA)
                        current_sign_placeholder.markdown('<p style="color: gray;">Waiting for sign...</p>', unsafe_allow_html=True)
                else:
                    # Speaking indicator
                    cv2.rectangle(frame, (0, 0), (W, 100), (255, 0, 0), -1)
                    cv2.putText(frame, "SPEAKING...", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                
                # Display current word and sentence
                current_word_str = ''.join(st.session_state.current_word)
                if current_word_str:
                    word_placeholder.markdown(f'<p class="word-display">{current_word_str}_</p>', unsafe_allow_html=True)
                else:
                    word_placeholder.markdown('<p class="word-display">(empty)</p>', unsafe_allow_html=True)
                
                sentence_str = ' '.join(st.session_state.sentence) if st.session_state.sentence else ''
                if sentence_str:
                    sentence_placeholder.markdown(f'<p class="sentence-display">{sentence_str}</p>', unsafe_allow_html=True)
                else:
                    sentence_placeholder.markdown('<p class="sentence-display">(empty)</p>', unsafe_allow_html=True)
                
                # Display frame with error handling
                if frame is not None and frame.size > 0 and H > 0 and W > 0:
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
            
            # Handle button clicks
            if speak_word_btn:
                if st.session_state.current_word:
                    word = ''.join(st.session_state.current_word)
                    st.session_state.sentence.append(word)
                    st.session_state.is_speaking = True
                    with audio_placeholder:
                        speak(word)
                    st.session_state.current_word = []
                    st.session_state.last_stable_char = None
                    st.session_state.is_speaking = False
                    status_text.success(f"🔊 Spoke: '{word}'")
                else:
                    status_text.warning("⚠️ No word to speak!")
            
            if delete_letter_btn:
                if st.session_state.current_word:
                    removed = st.session_state.current_word.pop()
                    st.session_state.last_stable_char = None
                    status_text.info(f"⌫ Deleted '{removed}'")
                else:
                    status_text.warning("⚠️ No letters to delete!")
            
            if clear_word_btn:
                if st.session_state.current_word:
                    cleared = ''.join(st.session_state.current_word)
                    st.session_state.current_word = []
                    st.session_state.last_stable_char = None
                    status_text.info(f"🗑️ Cleared word: '{cleared}'")
                else:
                    status_text.warning("⚠️ Word already empty!")
            
            if add_space_btn:
                st.session_state.sentence.append(' ')
                status_text.info("Added space")
            
            if reset_all_btn:
                st.session_state.current_word = []
                st.session_state.sentence = []
                st.session_state.last_stable_char = None
                status_text.info("🔄 Reset everything")
            
            if speak_sentence_btn:
                if st.session_state.sentence:
                    full_sentence = ' '.join(st.session_state.sentence)
                    st.session_state.is_speaking = True
                    with audio_placeholder:
                        speak(full_sentence)
                    st.session_state.is_speaking = False
                    status_text.success(f"🔊 Spoke sentence: '{full_sentence}'")
                elif st.session_state.current_word:
                    word = ''.join(st.session_state.current_word)
                    st.session_state.is_speaking = True
                    with audio_placeholder:
                        speak(word)
                    st.session_state.is_speaking = False
                    status_text.success(f"🔊 Spoke current word: '{word}'")
                else:
                    status_text.warning("⚠️ Nothing to speak!")
            
            # Small delay
            time.sleep(0.03)
            
            # Check if should stop (recheck the checkbox state)
            run = st.session_state.get('camera_toggle', False)
        
        cap.release()
        status_text.info("📷 Camera stopped")

# ===== SPEECH TO TEXT MODE =====
else:
    st.markdown("---")
    st.subheader("🎤 Speech to Sign Display")
    
    st.markdown("### ASL Alphabet Reference")
    st.info("💡 Type text below and we'll show you the corresponding ASL signs!")
    
    txt = st.text_input("Enter text to convert to ASL", placeholder="e.g., HELLO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔊 Speak Text"):
            if txt:
                speak(txt)
            else:
                st.warning("Please enter some text first!")
    
    with col2:
        if st.button("🤟 Show ASL Signs"):
            if txt:
                st.success(f"Showing ASL signs for: **{txt.upper()}**")
                
                st.markdown("### Sign Sequence:")
                for char in txt.upper():
                    if char.isalpha():
                        st.markdown(f"**{char}** → _(Display ASL sign image here)_")
                    elif char == ' ':
                        st.markdown("**[SPACE]**")
                
                st.info("📝 Note: In a full implementation, actual ASL sign images/animations would be displayed here.")
            else:
                st.warning("Please enter some text first!")
    
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Type any text in the input box above
    2. Click **Speak Text** to hear the text-to-speech output
    3. Click **Show ASL Signs** to see the corresponding ASL alphabet sequence
    4. For full implementation, integrate ASL sign images/GIFs for each letter
    """)

# Footer
st.markdown("---")
st.caption("Built for 24hr Hackathon | AI for Assistive & Human-Centric Technology")