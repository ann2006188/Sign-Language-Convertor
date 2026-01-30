# test_mediapipe.py
import sys
print(f"Python version: {sys.version}")

try:
    import mediapipe
    print(f"✓ MediaPipe version: {mediapipe.__version__}")
    print(f"✓ MediaPipe location: {mediapipe.__file__}")
    
    import mediapipe as mp
    print(f"✓ mp.solutions exists: {hasattr(mp, 'solutions')}")
    
    if hasattr(mp, 'solutions'):
        print(f"✓ mp.solutions.hands exists: {hasattr(mp.solutions, 'hands')}")
        hands = mp.solutions.hands
        print("✓ MediaPipe Hands initialized successfully!")
    else:
        print("✗ mp.solutions NOT FOUND - This is the problem")
        print(f"Available attributes: {dir(mp)}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()