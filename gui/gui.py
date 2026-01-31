import streamlit as st
from gtts import gTTS
import tempfile
import cv2
import numpy as np
import sys
import os
import time
import random
from pathlib import Path
import glob
from PIL import Image
import io

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    from sign_detector import SignDetector
    from sentence_builder import SentenceBuilder
    from transcriber import AudioTranscriber
except ImportError as e:
    st.error(f"❌ Could not import required modules: {e}")
    st.stop()

# Page config
st.set_page_config(
    layout="wide", 
    page_title="ASL Converter Pro", 
    page_icon="🤟",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL CSS STYLING
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Theme */
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --secondary: #10b981;
        --accent: #8b5cf6;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Headers */
    .app-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-bg);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-secondary);
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(37, 99, 235, 0.1);
    }
    
    /* Cards */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    
    /* Sentence Display */
    .sentence-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        background: var(--card-bg);
        border: 2px solid var(--primary);
        padding: 24px;
        border-radius: 12px;
        min-height: 80px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Text Preview */
    .text-preview {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        color: var(--text-primary);
        background: var(--card-bg);
        border: 1px solid var(--border);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        letter-spacing: 0.05em;
    }
    
    .highlight-current {
        color: var(--warning);
        font-weight: 700;
        font-size: 1.2em;
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        height: 48px;
        border-radius: 8px;
        border: none;
        background: var(--primary);
        color: white;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Text Areas */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        background-color: var(--card-bg);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Audio Recorder */
    .stAudioRecorder {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-live {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error);
        border: 1px solid var(--error);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Camera Container */
    .camera-container {
        border: 2px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Sign Image Container */
    .sign-image-container {
        background: var(--card-bg);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Sliders */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Section Headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_sign_image(letter, raw_data_path):
    """Get a random image for the given letter from raw_data"""
    if not letter.isalpha():
        return None
    
    letter = letter.upper()
    letter_folder = os.path.join(raw_data_path, letter)
    
    if not os.path.exists(letter_folder):
        return None
    
    extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
    images = []
    
    for ext in extensions:
        images.extend(glob.glob(os.path.join(letter_folder, ext)))
    
    if not images:
        return None
    
    return random.choice(images)

def generate_audio(text):
    """Generate audio from text using gTTS"""
    if not text.strip():
        return None
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            return f.name
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

# Path Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "raw_data")

# Validate raw_data path
if not os.path.exists(RAW_DATA_PATH):
    st.error(f"❌ Raw data folder not found at: {RAW_DATA_PATH}")
    st.info("Expected structure: Sign-language-convertor/raw_data/A, B, C, etc.")
    st.stop()

# Initialize session states
if 'builder' not in st.session_state:
    st.session_state.builder = SentenceBuilder()
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'sign_display_index' not in st.session_state:
    st.session_state.sign_display_index = 0
if 'is_displaying_signs' not in st.session_state:
    st.session_state.is_displaying_signs = False
if 'current_text_for_signs' not in st.session_state:
    st.session_state.current_text_for_signs = ""
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
# Separate state for Speech to Sign (Tab 4)
if 'sign_display_index_s2s' not in st.session_state:
    st.session_state.sign_display_index_s2s = 0
if 'is_displaying_signs_s2s' not in st.session_state:
    st.session_state.is_displaying_signs_s2s = False
if 'current_text_for_signs_s2s' not in st.session_state:
    st.session_state.current_text_for_signs_s2s = ""

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_detector():
    return SignDetector()

@st.cache_resource
def load_transcriber():
    return AudioTranscriber(model_size="medium")

detector = load_detector()
transcriber = load_transcriber()

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="app-header">🤟 ASL Converter Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Advanced Sign Language Translation System</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR SETTINGS
# ============================================
st.sidebar.markdown("## ⚙️ Settings")

# Detection Confidence
conf_threshold = st.sidebar.slider(
    "Detection Confidence", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Minimum confidence level for sign detection"
)

# Hold Time
hold_time_sec = st.sidebar.slider(
    "Sign Hold Time (sec)", 
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    help="Time to hold sign before detection"
)

# Display Duration
display_duration = st.sidebar.slider(
    "Sign Display Duration (sec)", 
    min_value=0.5, max_value=3.0, value=1.5, step=0.1,
    help="Duration to display each sign image"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")
st.sidebar.metric("Sentence Length", len(st.session_state.builder.sentence))
st.sidebar.metric("Transcribed Words", len(st.session_state.transcribed_text.split()))

# Update settings
st.session_state.builder.THRESHOLD = int(hold_time_sec * 30)
st.session_state.builder.REPEAT_THRESHOLD = int(hold_time_sec * 30) * 2
detector.hands.min_detection_confidence = conf_threshold

# ============================================
# TAB NAVIGATION
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Sign → Text", 
    "✍️ Text → Sign", 
    "🎤 Speech → Text",
    "🗣️ Speech → Sign"
])

# ============================================
# TAB 1: SIGN TO TEXT CONVERSION
# ============================================
with tab1:
    st.markdown('<p class="section-header">Sign Language to Text Conversion</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera Controls
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("▶️ Start Camera", use_container_width=True, key="start_cam_tab1"):
                st.session_state.camera_active = True
                st.rerun()
        with c2:
            if st.button("⏹️ Stop Camera", use_container_width=True, key="stop_cam_tab1"):
                st.session_state.camera_active = False
                st.rerun()
        with c3:
            if st.button("🔄 Convert to Sign", use_container_width=True, key="convert_tab1"):
                st.session_state.current_text_for_signs = st.session_state.builder.sentence
                st.session_state.sign_display_index = 0
                st.session_state.is_displaying_signs = True
                st.success("✓ Ready to display signs")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        stframe = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📝 Detected Text")
        
        # Status
        if st.session_state.camera_active:
            st.markdown(
                '<div class="status-badge status-live">'
                '<span class="status-dot"></span>Live'
                '</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-badge status-offline">'
                '<span class="status-dot"></span>Offline'
                '</div>', 
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Text Display
        text_placeholder = st.empty()
        sound_placeholder = st.empty()
        
        curr_sent = st.session_state.builder.sentence
        text_placeholder.markdown(
            f'<div class="sentence-display">{curr_sent if curr_sent else "Waiting for input..."}</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔊 Speak", use_container_width=True, key="speak_tab1"):
                if st.session_state.builder.sentence:
                    audio_file = generate_audio(st.session_state.builder.sentence)
                    if audio_file:
                        sound_placeholder.audio(audio_file, format='audio/mp3', autoplay=True)
                else:
                    st.warning("No text to speak")
        
        with col_btn2:
            if st.button("⌫ Backspace", use_container_width=True, key="back_tab1"):
                st.session_state.builder.sentence = st.session_state.builder.sentence[:-1]
                st.rerun()
        
        if st.button("🗑️ Clear All", use_container_width=True, key="clear_tab1"):
            st.session_state.builder.clear()
            st.rerun()
    
    # Camera Loop
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access camera")
            st.session_state.camera_active = False
        else:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                raw_char = detector.predict(frame)
                sentence, event = st.session_state.builder.process(raw_char)
                
                text_placeholder.markdown(
                    f'<div class="sentence-display">{sentence if sentence else "..."}</div>', 
                    unsafe_allow_html=True
                )
                
                if event == "SPEAK":
                    audio_file = generate_audio(sentence)
                    if audio_file:
                        sound_placeholder.audio(audio_file, format='audio/mp3', autoplay=True)
                
                H, W, _ = frame.shape
                if st.session_state.builder.temp_char and st.session_state.builder.frame_count > 0:
                    progress = st.session_state.builder.frame_count / st.session_state.builder.THRESHOLD
                    bar_w = int(W * min(progress, 1.0))
                    cv2.rectangle(frame, (0, H-15), (bar_w, H), (37, 99, 235), -1)
                
                if raw_char:
                    cv2.putText(frame, f"Detecting: {raw_char}", (20, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (37, 99, 235), 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01)
        
        cap.release()

# ============================================
# TAB 2: TEXT TO SIGN CONVERSION
# ============================================
with tab2:
    st.markdown('<p class="section-header">Text to Sign Language Conversion</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ✍️ Input Text")
        
        text_input = st.text_area(
            "Enter text to convert:",
            value=st.session_state.text_input,
            height=200,
            placeholder="Type your message here...",
            key="text_area_tab2"
        )
        st.session_state.text_input = text_input
        
        if st.button("🔄 Convert to Sign", use_container_width=True, key="convert_tab2"):
            if text_input.strip():
                st.session_state.current_text_for_signs = text_input.strip().upper()
                st.session_state.sign_display_index = 0
                st.session_state.is_displaying_signs = True
                st.success("✓ Starting conversion...")
                st.rerun()
            else:
                st.warning("Please enter text first")
        
        if st.button("🗑️ Clear Input", use_container_width=True, key="clear_tab2"):
            st.session_state.text_input = ""
            st.session_state.is_displaying_signs = False
            st.rerun()
    
    with col2:
        st.markdown("### 🖼️ Sign Display")
        
        sign_image_placeholder = st.empty()
        text_preview_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        if st.session_state.is_displaying_signs and st.session_state.current_text_for_signs:
            text = st.session_state.current_text_for_signs
            
            if st.session_state.sign_display_index < len(text):
                current_letter = text[st.session_state.sign_display_index]
                
                # Highlighted text
                highlighted_text = ""
                for i, char in enumerate(text):
                    if i == st.session_state.sign_display_index:
                        highlighted_text += f'<span class="highlight-current">{char}</span>'
                    else:
                        highlighted_text += char
                
                text_preview_placeholder.markdown(
                    f'<div class="text-preview">{highlighted_text}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sign
                if current_letter.isalpha():
                    img_path = get_sign_image(current_letter, RAW_DATA_PATH)
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            sign_image_placeholder.markdown(
                                f'<div class="sign-image-container"><h3>Sign: {current_letter}</h3></div>',
                                unsafe_allow_html=True
                            )
                            sign_image_placeholder.image(img, use_container_width=True)
                        except Exception as e:
                            sign_image_placeholder.error(f"Error loading image: {e}")
                    else:
                        sign_image_placeholder.warning(f"No image for '{current_letter}'")
                elif current_letter == ' ':
                    sign_image_placeholder.info("SPACE")
                else:
                    sign_image_placeholder.info(f"Special: '{current_letter}'")
                
                progress = (st.session_state.sign_display_index + 1) / len(text)
                progress_placeholder.progress(progress)
                
                time.sleep(display_duration)
                st.session_state.sign_display_index += 1
                st.rerun()
            else:
                text_preview_placeholder.success("✓ Conversion complete")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔁 Restart", use_container_width=True, key="restart_tab2"):
                        st.session_state.sign_display_index = 0
                        st.rerun()
                
                with col_b:
                    if st.button("🏠 Back", use_container_width=True, key="back_tab2"):
                        st.session_state.is_displaying_signs = False
                        st.rerun()
        else:
            sign_image_placeholder.info("Enter text and click 'Convert to Sign'")

# ============================================
# TAB 3: SPEECH TO TEXT
# ============================================
with tab3:
    st.markdown('<p class="section-header">Speech to Text Conversion</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎤 Audio Input")
        
        audio_input = st.audio_input("Record your speech:", key="audio_input_tab3")
        
        if audio_input is not None:
            st.audio(audio_input)
            
            if st.button("🔄 Transcribe Audio", use_container_width=True, key="transcribe_tab3"):
                with st.spinner("Transcribing..."):
                    # Convert audio to BytesIO
                    audio_bytes = io.BytesIO(audio_input.getvalue())
                    transcribed = transcriber.transcribe(audio_bytes)
                    st.session_state.transcribed_text = transcribed
                    st.success("✓ Transcription complete")
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Transcription", use_container_width=True, key="clear_tab3"):
            st.session_state.transcribed_text = ""
            st.rerun()
    
    with col2:
        st.markdown("### 📝 Transcribed Text")
        
        if st.session_state.transcribed_text:
            st.markdown(
                f'<div class="sentence-display">{st.session_state.transcribed_text}</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔊 Speak Text", use_container_width=True, key="speak_tab3"):
                    audio_file = generate_audio(st.session_state.transcribed_text)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3', autoplay=True)
            
            with col_b:
                if st.button("📋 Copy to Clipboard", use_container_width=True, key="copy_tab3"):
                    st.code(st.session_state.transcribed_text, language=None)
                    st.info("Text displayed above - use your browser to copy")
        else:
            st.info("Record audio and click 'Transcribe' to see results")

# ============================================
# TAB 4: SPEECH TO SIGN
# ============================================
with tab4:
    st.markdown('<p class="section-header">Speech to Sign Language Conversion</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎤 Audio Input")
        
        audio_input_s2s = st.audio_input("Record your speech:", key="audio_input_tab4")
        
        if audio_input_s2s is not None:
            st.audio(audio_input_s2s)
            
            if st.button("🔄 Convert to Sign", use_container_width=True, key="convert_tab4"):
                with st.spinner("Processing..."):
                    # Transcribe
                    audio_bytes = io.BytesIO(audio_input_s2s.getvalue())
                    transcribed = transcriber.transcribe(audio_bytes)
                    
                    if transcribed:
                        st.session_state.current_text_for_signs_s2s = transcribed.upper()
                        st.session_state.sign_display_index_s2s = 0
                        st.session_state.is_displaying_signs_s2s = True
                        st.success("✓ Starting sign display")
                        st.rerun()
                    else:
                        st.error("Transcription failed")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All", use_container_width=True, key="clear_tab4"):
            st.session_state.is_displaying_signs_s2s = False
            st.session_state.current_text_for_signs_s2s = ""
            st.session_state.sign_display_index_s2s = 0
            st.rerun()
    
    with col2:
        st.markdown("### 🖼️ Sign Display")
        
        sign_image_placeholder_s2s = st.empty()
        text_preview_placeholder_s2s = st.empty()
        progress_placeholder_s2s = st.empty()
        
        if st.session_state.is_displaying_signs_s2s and st.session_state.current_text_for_signs_s2s:
            text = st.session_state.current_text_for_signs_s2s
            
            if st.session_state.sign_display_index_s2s < len(text):
                current_letter = text[st.session_state.sign_display_index_s2s]
                
                # Highlighted text
                highlighted_text = ""
                for i, char in enumerate(text):
                    if i == st.session_state.sign_display_index_s2s:
                        highlighted_text += f'<span class="highlight-current">{char}</span>'
                    else:
                        highlighted_text += char
                
                text_preview_placeholder_s2s.markdown(
                    f'<div class="text-preview">{highlighted_text}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sign
                if current_letter.isalpha():
                    img_path = get_sign_image(current_letter, RAW_DATA_PATH)
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            sign_image_placeholder_s2s.markdown(
                                f'<div class="sign-image-container"><h3>Sign: {current_letter}</h3></div>',
                                unsafe_allow_html=True
                            )
                            sign_image_placeholder_s2s.image(img, use_container_width=True)
                        except Exception as e:
                            sign_image_placeholder_s2s.error(f"Error: {e}")
                    else:
                        sign_image_placeholder_s2s.warning(f"No image for '{current_letter}'")
                elif current_letter == ' ':
                    sign_image_placeholder_s2s.info("SPACE")
                else:
                    sign_image_placeholder_s2s.info(f"Special: '{current_letter}'")
                
                progress = (st.session_state.sign_display_index_s2s + 1) / len(text)
                progress_placeholder_s2s.progress(progress)
                
                time.sleep(display_duration)
                st.session_state.sign_display_index_s2s += 1
                st.rerun()
            else:
                text_preview_placeholder_s2s.success("✓ Conversion complete")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔁 Restart", use_container_width=True, key="restart_tab4"):
                        st.session_state.sign_display_index_s2s = 0
                        st.rerun()
                
                with col_b:
                    if st.button("🏠 Reset", use_container_width=True, key="reset_tab4"):
                        st.session_state.is_displaying_signs_s2s = False
                        st.session_state.current_text_for_signs_s2s = ""
                        st.rerun()
        else:
            sign_image_placeholder_s2s.info("Record audio and click 'Convert to Sign'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-family: Inter, sans-serif; font-size: 0.875rem;'>"
    "ASL Converter Pro © 2026 | Powered by MediaPipe, Whisper & Streamlit"
    "</p>",
    unsafe_allow_html=True
)