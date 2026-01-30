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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    from sign_detector import SignDetector
    from sentence_builder import SentenceBuilder
except ImportError:
    st.error("❌ Could not import 'src' modules.")
    st.stop()

# Page config
st.set_page_config(
    layout="wide", 
    page_title="ASL Converter Pro", 
    page_icon="🤟",
    initial_sidebar_state="expanded"
)

# ============================================
# MODERN CSS STYLING
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global Theme */
    :root {
        --primary-glow: #00ffff;
        --secondary-glow: #ff00ff;
        --dark-bg: #0a0a0a;
        --card-bg: #151515;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Custom Headers */
    .app-header {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(135deg, #00ffff, #ff00ff, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,255,255,0.5);
        margin-bottom: 2rem;
        animation: glow-pulse 3s ease-in-out infinite;
    }
    
    @keyframes glow-pulse {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(0,255,255,0.7)); }
        50% { filter: drop-shadow(0 0 25px rgba(255,0,255,0.9)); }
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: rgba(21, 21, 21, 0.8);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-secondary);
        background-color: transparent;
        border: 2px solid transparent;
        border-radius: 10px;
        padding: 12px 25px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2));
        border: 2px solid var(--primary-glow);
        color: var(--primary-glow);
        box-shadow: 0 0 20px rgba(0,255,255,0.4);
    }
    
    /* Sentence Display */
    .sentence-display {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-glow);
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
        padding: 25px;
        border-radius: 15px;
        border: 2px solid var(--primary-glow);
        min-height: 100px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,255,255,0.3);
        backdrop-filter: blur(10px);
        animation: border-glow 2s ease-in-out infinite;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    @keyframes border-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0,255,255,0.3); }
        50% { box-shadow: 0 0 40px rgba(255,0,255,0.5); }
    }
    
    /* Text Input Area */
    .stTextArea textarea {
        font-family: 'Space Mono', monospace;
        font-size: 1.2rem;
        background-color: rgba(21, 21, 21, 0.9);
        color: var(--text-primary);
        border: 2px solid var(--primary-glow);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        height: 55px;
        border-radius: 12px;
        border: 2px solid var(--primary-glow);
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
        color: var(--primary-glow);
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0,255,255,0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(0,255,255,0.3), rgba(255,0,255,0.3));
        box-shadow: 0 0 30px rgba(0,255,255,0.6);
        transform: translateY(-2px);
    }
    
    /* Sign Image Display */
    .sign-image-container {
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
        border: 3px solid var(--primary-glow);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 0 40px rgba(0,255,255,0.4);
        text-align: center;
        animation: image-pulse 2s ease-in-out infinite;
    }
    
    @keyframes image-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Highlighted Text */
    .highlight-current {
        color: #ffff00;
        font-weight: 900;
        text-shadow: 0 0 10px rgba(255,255,0,0.8);
        font-size: 1.3em;
    }
    
    .text-preview {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        color: var(--text-primary);
        background: rgba(21, 21, 21, 0.8);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid var(--secondary-glow);
        margin: 20px 0;
        text-align: center;
        letter-spacing: 0.1em;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
        border-right: 2px solid var(--primary-glow);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Orbitron', sans-serif;
        color: var(--primary-glow);
        text-shadow: 0 0 15px rgba(0,255,255,0.6);
    }
    
    /* Camera Feed Border */
    .camera-container {
        border: 3px solid var(--primary-glow);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(0,255,255,0.4);
    }
    
    /* Status Indicators */
    .status-live {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #00ff00;
        border-radius: 50%;
        animation: blink 1s ease-in-out infinite;
        margin-right: 8px;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Sliders */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Progress Elements */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-glow), var(--secondary-glow));
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_sign_image(letter, raw_data_path):
    """Get a random image for the given letter from raw_data"""
    # Handle only alphabetic characters
    if not letter.isalpha():
        return None
    
    letter = letter.upper()
    letter_folder = os.path.join(raw_data_path, letter)
    
    # Debug: Check if folder exists
    if not os.path.exists(letter_folder):
        st.warning(f"⚠️ Folder not found: {letter_folder}")
        return None
    
    # Try multiple extensions with case variations
    extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]
    images = []
    
    for ext in extensions:
        images.extend(glob.glob(os.path.join(letter_folder, ext)))
    
    # Debug: Show what we found
    if not images:
        st.warning(f"⚠️ No images found in: {letter_folder}")
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

# Path Configuration - Get raw_data path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from gui/ to project root
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "raw_data")

# Validate raw_data path exists
if not os.path.exists(RAW_DATA_PATH):
    st.error(f"❌ Raw data folder not found at: {RAW_DATA_PATH}")
    st.info("Expected structure: Sign-language-convertor/raw_data/A, B, C, etc.")
    st.stop()

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

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="app-header">🤟 ASL CONVERTER PRO</h1>', unsafe_allow_html=True)

# ============================================
# SIDEBAR SETTINGS
# ============================================
st.sidebar.markdown("## ⚙️ SETTINGS")

# Confidence Threshold
conf_threshold = st.sidebar.slider(
    "🎯 Detection Confidence", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

# Hold Time for Sign Detection
hold_time_sec = st.sidebar.slider(
    "⏱️ Sign Hold Time (sec)", 
    min_value=0.1, max_value=2.0, value=0.5, step=0.1
)

# Display Duration for Text-to-Sign
display_duration = st.sidebar.slider(
    "🖼️ Sign Display Duration (sec)", 
    min_value=0.5, max_value=3.0, value=1.5, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 STATISTICS")
st.sidebar.metric("Current Sentence Length", len(st.session_state.builder.sentence))

# Update Builder Logic
st.session_state.builder.THRESHOLD = int(hold_time_sec * 30)
st.session_state.builder.REPEAT_THRESHOLD = int(hold_time_sec * 30) * 2

# Load Detector
@st.cache_resource
def load_detector():
    return SignDetector()

detector = load_detector()
detector.hands.min_detection_confidence = conf_threshold

# ============================================
# TAB NAVIGATION
# ============================================
tab1, tab2 = st.tabs(["📹 SIGN → TEXT", "✍️ TEXT → SIGN"])

# ============================================
# TAB 1: SIGN TO TEXT CONVERSION
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 LIVE CAMERA FEED")
        
        # Camera Controls
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            start_btn = st.button("▶️ START CAMERA", use_container_width=True)
            if start_btn:
                st.session_state.camera_active = True
                st.rerun()
        with c2:
            stop_btn = st.button("⏹️ STOP CAMERA", use_container_width=True)
            if stop_btn:
                st.session_state.camera_active = False
                st.rerun()
        with c3:
            convert_to_sign_btn = st.button("🔄 CONVERT TO SIGN", use_container_width=True)
            if convert_to_sign_btn:
                # Transfer current sentence to text-to-sign
                st.session_state.current_text_for_signs = st.session_state.builder.sentence
                st.session_state.sign_display_index = 0
                st.session_state.is_displaying_signs = True
                st.success("✅ Switched to TEXT → SIGN mode!")
                time.sleep(1)
                st.rerun()
        
        # Camera Feed Container
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        stframe = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📝 DETECTED TEXT")
        
        # Status Indicator
        if st.session_state.camera_active:
            st.markdown('<span class="status-live"></span><b>LIVE</b>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #ff4444;">●</span> <b>OFFLINE</b>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Text Display Placeholder
        text_placeholder = st.empty()
        sound_placeholder = st.empty()
        
        # Initial Render
        curr_sent = st.session_state.builder.sentence
        text_placeholder.markdown(
            f'<div class="sentence-display">{curr_sent if curr_sent else "Waiting for input..."}</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔊 SPEAK", use_container_width=True):
                audio_file = generate_audio(st.session_state.builder.sentence)
                if audio_file:
                    sound_placeholder.audio(audio_file, format='audio/mp3', autoplay=True)
                else:
                    st.warning("No text to speak")
        
        with col_btn2:
            if st.button("⌫ BACKSPACE", use_container_width=True):
                st.session_state.builder.sentence = st.session_state.builder.sentence[:-1]
                st.rerun()
        
        if st.button("🗑️ CLEAR ALL", use_container_width=True):
            st.session_state.builder.clear()
            st.rerun()
    
    # ============================================
    # CAMERA LOOP
    # ============================================
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not open camera. Check permissions.")
            st.session_state.camera_active = False
        else:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Prediction
                raw_char = detector.predict(frame)
                
                # Process with SentenceBuilder
                sentence, event = st.session_state.builder.process(raw_char)
                
                # Update UI
                text_placeholder.markdown(
                    f'<div class="sentence-display">{sentence if sentence else "..."}</div>', 
                    unsafe_allow_html=True
                )
                
                # Auto-speak on event
                if event == "SPEAK":
                    audio_file = generate_audio(sentence)
                    if audio_file:
                        sound_placeholder.audio(audio_file, format='audio/mp3', autoplay=True)
                
                # Visual Progress Bar
                H, W, _ = frame.shape
                if st.session_state.builder.temp_char and st.session_state.builder.frame_count > 0:
                    progress = st.session_state.builder.frame_count / st.session_state.builder.THRESHOLD
                    bar_w = int(W * min(progress, 1.0))
                    cv2.rectangle(frame, (0, H-20), (bar_w, H), (0, 255, 255), -1)
                
                # Draw detected letter
                if raw_char:
                    cv2.putText(
                        frame, f"SCAN: {raw_char}", 
                        (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 255), 3
                    )
                
                # Display frame
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Small delay for UI responsiveness
                time.sleep(0.01)
        
        cap.release()

# ============================================
# TAB 2: TEXT TO SIGN CONVERSION
# ============================================
with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ✍️ INPUT TEXT")
        
        # Text Input
        text_input = st.text_area(
            "Enter text to convert to sign language:",
            value=st.session_state.text_input,
            height=200,
            key="text_area_input"
        )
        st.session_state.text_input = text_input
        
        # Convert Button
        if st.button("🔄 CONVERT TO SIGN", use_container_width=True, key="convert_btn"):
            if text_input.strip():
                st.session_state.current_text_for_signs = text_input.strip().upper()
                st.session_state.sign_display_index = 0
                st.session_state.is_displaying_signs = True
                st.success("✅ Starting sign display...")
                st.rerun()
            else:
                st.warning("⚠️ Please enter some text first!")
        
        # Clear Button
        if st.button("🗑️ CLEAR INPUT", use_container_width=True, key="clear_input_btn"):
            st.session_state.text_input = ""
            st.session_state.is_displaying_signs = False
            st.rerun()
    
    with col2:
        st.markdown("### 🖼️ SIGN LANGUAGE DISPLAY")
        
        # Debug: Show current paths
        with st.expander("🔍 Debug Info"):
            st.write(f"**Raw Data Path:** `{RAW_DATA_PATH}`")
            st.write(f"**Path Exists:** {os.path.exists(RAW_DATA_PATH)}")
            if os.path.exists(RAW_DATA_PATH):
                folders = [f for f in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, f))]
                st.write(f"**Available Letter Folders:** {', '.join(sorted(folders))}")
        
        # Display Container
        sign_image_placeholder = st.empty()
        text_preview_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        if st.session_state.is_displaying_signs and st.session_state.current_text_for_signs:
            text = st.session_state.current_text_for_signs
            
            # Display with highlighting
            if st.session_state.sign_display_index < len(text):
                current_letter = text[st.session_state.sign_display_index]
                
                # Build highlighted HTML
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
                
                # Get and display sign image
                if current_letter.isalpha():
                    img_path = get_sign_image(current_letter, RAW_DATA_PATH)
                    
                    if img_path and os.path.exists(img_path):
                        # Use PIL to load and display image for better compatibility
                        try:
                            from PIL import Image
                            img = Image.open(img_path)
                            
                            sign_image_placeholder.markdown(
                                f'<div class="sign-image-container"><h2>Sign for: {current_letter}</h2></div>',
                                unsafe_allow_html=True
                            )
                            sign_image_placeholder.image(
                                img, 
                                use_container_width=True
                            )
                        except Exception as e:
                            sign_image_placeholder.error(f"❌ Error loading image: {e}")
                            sign_image_placeholder.write(f"Path: {img_path}")
                    else:
                        sign_image_placeholder.warning(f"⚠️ No image found for '{current_letter}'")
                        sign_image_placeholder.write(f"Looking in: {os.path.join(RAW_DATA_PATH, current_letter)}")
                elif current_letter == ' ':
                    # Handle spaces
                    sign_image_placeholder.info(f"📝 SPACE")
                else:
                    # Handle special characters
                    sign_image_placeholder.info(f"📝 Special character: '{current_letter}'")
                
                # Progress bar
                progress = (st.session_state.sign_display_index + 1) / len(text)
                progress_placeholder.progress(progress)
                
                # Auto-advance
                time.sleep(display_duration)
                st.session_state.sign_display_index += 1
                st.rerun()
            else:
                # Completed
                text_preview_placeholder.success("✅ Sign language conversion complete!")
                sign_image_placeholder.balloons()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔁 RESTART", use_container_width=True):
                        st.session_state.sign_display_index = 0
                        st.rerun()
                
                with col_b:
                    if st.button("🏠 BACK TO INPUT", use_container_width=True):
                        st.session_state.is_displaying_signs = False
                        st.rerun()
        else:
            # Default state
            sign_image_placeholder.info("👈 Enter text and click 'CONVERT TO SIGN' to begin")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-family: Space Mono, monospace;'>"
    "🤟 ASL Converter Pro | Powered by MediaPipe & Streamlit | 2024"
    "</p>",
    unsafe_allow_html=True
)