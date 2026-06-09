# Sign Language Assistive Translator

An assistive, human-centric AI system that enables real-time communication
between sign language users and non-sign language users by translating hand
gestures into text and speech.

Built as part of a university NNDL (Neural Networks and Deep Learning) project,
upgraded from a classical ML pipeline to a full PyTorch deep learning backbone.

---

## Features
- Real-time sign language gesture recognition via webcam
- Sign → Text translation with majority-vote smoothing
- Text → Speech output for accessibility (pyttsx3)
- Lightweight and fully offline-capable
- Human-centric, low-cost design using only open-source components

---

## System Overview
The system uses **MediaPipe Hand Landmarker** (Tasks API) to extract 21 3-D
hand landmarks from each webcam frame. The 42 normalised (x, y) coordinates are
then *rendered* as a white skeletal line-drawing on a 224 × 224 black canvas.
This synthesised image is fed into a pretrained **ResNet-18** or
**EfficientNet-B0** CNN backbone (transfer learning via PyTorch / torchvision).
Only the final classification head is fine-tuned to the gesture classes.
The predicted class index is mapped back to the sign-language label, displayed
on-screen, and—optionally—spoken aloud via pyttsx3.

---

## Tech Stack
| Component | Technology |
|:---|:---|
| Programming Language | Python 3.13 |
| Computer Vision / Landmark Extraction | MediaPipe 0.10.33 (Tasks API), OpenCV |
| Deep Learning Framework | PyTorch 2.11.0 |
| Model Backbone (Transfer Learning) | ResNet-18, EfficientNet-B0 (torchvision 0.26.0) |
| Loss Function | CrossEntropyLoss |
| Optimiser | Adam |
| Classical ML (legacy) | Scikit-Learn |
| Text-to-Speech | pyttsx3 |
| Speech-to-Text (optional) | SpeechRecognition |
| Frontend / UI | Streamlit |

All components are open-source and require no paid APIs.

---

## Project Structure
```
Sign-Language-Convertor/
├── gui/                  # Streamlit user interface
├── data/                 # Processed landmark CSV files
├── model/                # Saved .pth model weights + class list
├── plots/                # Training curves, confusion matrix, ROC curve
├── src/
│   ├── train_model.py    # PyTorch training — 3 experiments, full metrics
│   ├── sign_detector.py  # Real-time inference class (MediaPipe + PyTorch)
│   ├── inference.py      # CLI webcam demo
│   ├── collect_custom.py # Webcam data collection tool
│   ├── process_images.py # Batch image → landmark CSV converter
│   ├── fix_data.py       # Landmark normalisation
│   ├── sentence_builder.py
│   └── transcriber.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (3 experiments, 10 epochs each)
```bash
python src/train_model.py
```
Outputs:
- `model/sign_dl_model.pth` — best model weights
- `plots/training_curves.png` — loss & accuracy vs epochs
- `plots/confusion_matrix.png`
- `plots/roc_curve.png`
- `results_table.md` — experiment comparison table

### 3. Run the live webcam demo
```bash
python src/inference.py
```
Press **Q** or **Esc** to quit.
