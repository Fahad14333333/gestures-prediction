#!/usr/bin/env python3
"""
Real-time sign language interpretation assistant.
Uses computer vision to detect hand gestures from camera and speaks the interpretation.
"""

import cv2
import mediapipe as mp
import pyttsx3
import time

# Mapping from raw vision model labels to gesture names (for compatibility)
RAW_TO_GESTURE = {
    "THUMB_UP": "YES",
    "POINT_SELF": "ME"
}

# Mapping of gestures to their meanings and speech outputs
GESTURES = {
    "ME": ("Self reference", "I"),
    "YOU": ("Referring to you", "You"),
    "LOVE": ("Affection", "I love you"),
    "YES": ("Agreement", "Yes"),
    "NO": ("Disagreement", "No"),
    "HELP": ("Request for assistance", "Help"),
    "HAPPY": ("Joy", "I'm happy"),
    "SAD": ("Sorrow", "I'm sad"),
    "THANK YOU": ("Gratitude", "Thank you"),
    "FRIEND": ("Friendship", "Friend")
}

def interpret_gesture(gesture):
    """
    Interpret a detected gesture label.
    
    Args:
        gesture (str): The detected gesture label
        
    Returns:
        tuple: (meaning, speech) or None if unknown
    """
    gesture = gesture.strip().upper()
    
    # Map raw labels to gesture names if available
    if gesture in RAW_TO_GESTURE:
        gesture = RAW_TO_GESTURE[gesture]
    
    if gesture in GESTURES:
        return GESTURES[gesture]
    return None

def classify_gesture(hand_landmarks):
    """
    Classify gesture based on MediaPipe hand landmarks.
    This is a simple heuristic-based classifier.
    
    Args:
        hand_landmarks: List of MediaPipe NormalizedLandmark objects
        
    Returns:
        str: Detected gesture or None
    """
    # Extract landmark positions
    landmarks = []
    for lm in hand_landmarks:
        landmarks.append((lm.x, lm.y, lm.z))
    
    # Helper functions
    def is_finger_extended(tip, pip, mcp):
        """Check if finger is extended"""
        return landmarks[tip][1] < landmarks[pip][1] and landmarks[pip][1] < landmarks[mcp][1]
    
    def is_finger_curled(tip, pip, mcp):
        """Check if finger is curled"""
        return landmarks[tip][1] > landmarks[pip][1]
    
    # Thumb (landmarks 4, 3, 2)
    thumb_extended = landmarks[4][0] > landmarks[3][0]  # Thumb pointing right (for right hand)
    
    # Index finger (8, 7, 6)
    index_extended = is_finger_extended(8, 7, 6)
    
    # Middle finger (12, 11, 10)
    middle_extended = is_finger_extended(12, 11, 10)
    
    # Ring finger (16, 15, 14)
    ring_extended = is_finger_extended(16, 15, 14)
    
    # Pinky (20, 19, 18)
    pinky_extended = is_finger_extended(20, 19, 18)
    
    # Simple gesture detection
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "YES"  # Thumb up
    
    if index_extended and not thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "ME"  # Pointing (simplified)
    
    if index_extended and middle_extended and not thumb_extended and not ring_extended and not pinky_extended:
        return "LOVE"  # V sign for love
    
    if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "NO"  # Fist for no
    
    # Add more gestures as needed
    # For now, return None for unknown
    return None

def main():
    # Initialize MediaPipe Hands using tasks API
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Create hand landmarker with downloaded model
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    try:
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing hand landmarker: {e}")
        print("Using simplified fallback...")
        return
    
    # Initialize text-to-speech
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting real-time gesture detection. Press 'q' to quit.")
    
    last_gesture = None
    last_speak_time = 0
    speak_cooldown = 2  # seconds between speeches
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect hand landmarks
        result = hand_landmarker.detect(mp_image)
        
        current_gesture = None
        
        if result.hand_landmarks:
            # Take the first hand
            hand_landmarks = result.hand_landmarks[0]
            
            # Classify gesture
            current_gesture = classify_gesture(hand_landmarks)
            
            if current_gesture:
                # Display detected gesture
                cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Speak if gesture detected and cooldown passed
        if current_gesture and current_gesture != last_gesture and time.time() - last_speak_time > speak_cooldown:
            interpretation = interpret_gesture(current_gesture)
            if interpretation:
                meaning, speech = interpretation
                print(f"{current_gesture} → Meaning: {meaning} → Speech: {speech}")
                engine.say(speech)
                engine.runAndWait()
                last_gesture = current_gesture
                last_speak_time = time.time()
        
        # Display frame
        cv2.imshow('Gesture Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

if __name__ == "__main__":
    main()