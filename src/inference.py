import sys
import os
import cv2
import time
import numpy as np

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_detector import SignDetector

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading Deep Learning Model...")
try:
    detector = SignDetector()
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Run 'python src/train_model.py' first to generate the model files.")
    sys.exit(1)
print("Model loaded successfully!\n")

# ── Open webcam — try every combination of index + backend ───────────────────
def try_open_camera():
    """Try camera indices 0 and 1 with multiple backends. Returns a working cap."""
    candidates = [
        (0, cv2.CAP_MSMF,    "index 0 / MSMF"),
        (0, cv2.CAP_DSHOW,   "index 0 / DirectShow"),
        (0, cv2.CAP_ANY,     "index 0 / default"),
        (1, cv2.CAP_MSMF,    "index 1 / MSMF"),
        (1, cv2.CAP_DSHOW,   "index 1 / DirectShow"),
        (1, cv2.CAP_ANY,     "index 1 / default"),
    ]
    for idx, backend, label in candidates:
        print(f"  Trying camera {label}...", end=" ", flush=True)
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print("could not open.")
            continue

        # Give the sensor time to start streaming
        time.sleep(0.5)

        # Try to grab a real (non-black) frame
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.any():
                print("OK!")
                return cap, label
            time.sleep(0.1)

        print("opens but returns black frames.")
        cap.release()

    return None, None


print("Searching for a working camera...")
cap, cam_label = try_open_camera()

if cap is None:
    print("\nERROR: No working camera found. Make sure a webcam is connected and "
          "not used by another application (Teams, Zoom, Camera app, etc.).")
    sys.exit(1)

print(f"Using camera: {cam_label}\n")

# Set stable resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

# One more warm-up pass after setting properties
time.sleep(0.3)
for _ in range(10):
    cap.read()

print("Press 'Q' or Esc to quit.\n")

# ── State ────────────────────────────────────────────────────────────────────
history   = []
SMOOTH_N  = 5
predicted_char = "..."
fail_count = 0
MAX_FAILS  = 30   # after 30 consecutive bad reads, give up

# ── Main loop ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret or frame is None or not frame.any():
        fail_count += 1
        if fail_count >= MAX_FAILS:
            print("ERROR: Camera stopped sending frames. Exiting.")
            break
        time.sleep(0.05)
        continue

    fail_count = 0
    frame = cv2.flip(frame, 1)   # mirror

    # ── Gesture prediction ───────────────────────────────────────────────────
    raw_pred = detector.predict(frame)   # also draws landmarks onto frame

    if raw_pred is not None:
        history.append(raw_pred)
        if len(history) > SMOOTH_N:
            history.pop(0)
        predicted_char = max(set(history), key=history.count)
    else:
        history.clear()
        predicted_char = "..."

    # ── HUD overlay ─────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Sign Language Detector", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    label_color = (0, 255, 120) if predicted_char != "..." else (120, 120, 120)
    cv2.putText(frame, predicted_char, (10, 70),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, label_color, 3, cv2.LINE_AA)

    cv2.putText(frame, "Q / Esc to quit", (w - 200, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Closed.")