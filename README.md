# Gestures Prediction

A real-time sign language interpretation assistant that uses computer vision to detect hand gestures from your camera and converts them into spoken interpretations.

## Features

- Real-time hand gesture detection using camera
- Supports 10 common sign language gestures
- Text-to-speech output for immediate feedback
- Visual feedback with hand landmark display

## Supported Gestures

The assistant recognizes and interprets the following gestures:

- **ME**: Pointing with index finger (self reference)
- **YOU**: Referring to you
- **LOVE**: V-sign with index and middle fingers (affection)
- **YES**: Thumb up (agreement)
- **NO**: Closed fist (disagreement)
- **HELP**: Request for assistance
- **HAPPY**: Joy
- **SAD**: Sorrow
- **THANK YOU**: Gratitude
- **FRIEND**: Friendship

## Requirements

- Python 3.x
- Webcam/camera
- Dependencies: opencv-python, mediapipe, pyttsx3

## Installation

```bash
pip install opencv-python mediapipe pyttsx3
```

## Usage

Run the real-time gesture detection:

```bash
python3 gesture_interpreter.py
```

Or if using a virtual environment:

```bash
.venv-1/bin/python gesture_interpreter.py
```

- The camera will open showing your hand with landmarks
- Make gestures in front of the camera
- Detected gestures will be spoken aloud
- Press 'q' to quit

## Gesture Recognition

The system uses MediaPipe for hand tracking and simple heuristics to classify gestures based on finger positions. For best results:

- Ensure good lighting
- Keep your hand clearly visible
- Hold gestures steady for 1-2 seconds
- Use right hand (left hand support can be added)

## Output Format

When a gesture is detected, you'll see:
- Console output: `GESTURE → Meaning: DESCRIPTION → Speech: SENTENCE`
- Audio output: The spoken interpretation

## Examples

- Thumb up → "YES → Meaning: Agreement → Speech: Yes"
- Index finger pointing → "ME → Meaning: Self reference → Speech: I"
- V-sign → "LOVE → Meaning: Affection → Speech: I love you"

## Technical Details

- Uses MediaPipe Hands for robust hand detection
- Heuristic-based gesture classification (no ML model required)
- Pyttsx3 for cross-platform text-to-speech
- OpenCV for camera handling and display