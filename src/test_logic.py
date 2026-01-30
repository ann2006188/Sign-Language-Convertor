import cv2
from sign_detector import SignDetector
from sentence_builder import SentenceBuilder

# Initialize Eyes and Brain
detector = SignDetector()
builder = SentenceBuilder()

cap = cv2.VideoCapture(0)

print("Started! Try spelling 'HELLO'. Use 'del' to fix mistakes.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # 1. Get Raw Prediction (The "Eyes")
    raw_char = detector.predict(frame)

    # 2. Process Logic (The "Brain")
    sentence, event = builder.process(raw_char)

    # 3. Feedback Logic
    if event == "SPEAK":
        print(f"\n[EVENT] Speech Triggered! Final Sentence: {sentence}\n")

    # Display on screen
    # Show Raw Prediction (Top Left - Small)
    cv2.putText(frame, f"Raw: {raw_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show Loading Bar for Stability (Visual Feedback)
    if builder.temp_char and builder.frame_count > 0:
        bar_width = int((builder.frame_count / builder.THRESHOLD) * 100)
        cv2.rectangle(frame, (10, 40), (10 + bar_width, 50), (255, 255, 0), -1)

    # Show Built Sentence (Bottom - Big)
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, sentence, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow('Logic Tester', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()