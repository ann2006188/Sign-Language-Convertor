import whisper
import os
import tempfile
import torch

class AudioTranscriber:
    def __init__(self, model_size="medium"):
        """
        Loads the Whisper model on GPU if available.
        """
        # 1. Detect Device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.fp16 = True
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = "cpu"
            self.fp16 = False # CPU doesn't support fp16 operations well
            print("⚠️ GPU not found. Falling back to CPU (Slower).")

        print(f"Loading Whisper model ({model_size}) on {self.device}...")
        
        try:
            # 2. Load Model onto Specific Device
            self.model = whisper.load_model(model_size, device=self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            self.model = None

    def transcribe(self, audio_file_obj):
        """
        Input: Streamlit Audio Data (BytesIO)
        Output: Transcribed Text (str)
        """
        if self.model is None:
            return "Error: Model not loaded."

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file_obj.read())
            tmp_path = tmp_file.name

        try:
            # 3. Transcribe with GPU optimizations
            result = self.model.transcribe(tmp_path, fp16=self.fp16)
            text = result["text"].strip()
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return text
            
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return f"Error during transcription: {e}"