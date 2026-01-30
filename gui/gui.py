import streamlit as st
from gtts import gTTS
import tempfile

st.set_page_config(layout="wide")

st.title("ASL Convertor")
st.caption("Sign to text to speech")

mode = st.sidebar.radio(
    "mode",
    ["sign → text → speech", "speech → text"]
)

def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        st.audio(f.name)

def fake_predict(img):
    return "hello", 0.92

if mode == "sign → text → speech":
    img = st.camera_input("show your hand")
    if img:
        text, conf = fake_predict(img)
        st.success(f"{text} ({conf*100:.1f}%)")
        speak(text)

else:
    txt = st.text_input("enter text")
    if st.button("speak"):
        speak(txt)
