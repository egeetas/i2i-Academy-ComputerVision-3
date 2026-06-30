# i2i Academy - Computer Vision Assignment (Homework 1)

This repository contains the complete implementation and theoretical answers for the i2i Academy Computer Vision homework. The goal of this assignment is to develop a real-time finger-counting application using OpenCV and MediaPipe.

---

## Section 3.1.1: Theoretical Knowledge Answers

Here are the answers to the three theoretical questions (constrained to a maximum of 5 sentences in total):

1. **Computer Vision (CV)** is a field of artificial intelligence that enables computers to interpret and understand digital images or videos, with primary use cases including autonomous driving, medical image analysis, facial recognition, and industrial quality control.
2. The key difference is that **image classification** assigns a single label to an entire image, whereas **object detection** identifies, locates, and draws bounding boxes around multiple individual objects within that image.
3. Software engineers prefer pre-trained frameworks like **MediaPipe** because they provide highly optimized, real-time, cross-platform performance out of the box, saving significant time, compute resources, and data labeling effort compared to training a neural network from scratch.

---

## Section 3.1.2: Practical Application Details

The practical application is written in Python (`hand_tracking.py`). It is implemented using modern **Object-Oriented Design Patterns** to achieve high modularity and clean separation of concerns:

- **Strategy Pattern**: Evaluates finger open/closed states using interchangeable algorithms:
  - `StandardFinger` strategy compares the y-coordinate of the tip with the PIP joint (tip y < PIP y is open).
  - `Thumb` strategy evaluates horizontal x-coordinate limits dynamically based on hand orientation (comparing Pinky MCP and Thumb MCP x-coordinates).
- **Factory Pattern**: `FingerFactory` dynamically instantiates `Finger` strategy subclasses with appropriate configurations and landmark indices.
- **Facade Pattern**: `HandTracker` wraps MediaPipe Hands initialization, process, and landmark drawing to expose a simplified tracking API.
- **Composition Pattern**: `Hand` represents a hand composed of a list of `Finger` objects, delegating state evaluation to individual fingers.
- **Application Controller**: `FingerCounterApp` manages OpenCV webcam capture, frame processing, and HUD graphics output.

---

## Quick Start & Setup Guide

### 1. Prerequisites
Ensure you have Python 3 installed on your machine.

### 2. Setup a Virtual Environment (Recommended)
Open your terminal in this directory and run:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
Install OpenCV and MediaPipe:
```bash
pip install -r requirements.txt
```

### 4. Run the Finger Counter
Run the application:
```bash
python hand_tracking.py
```

### 5. Control Instructions
- Hold up one or two hands in front of the camera.
- The screen will display the detected landmarks, individual finger states (OPEN/CLOSED), and the total open finger count.
- Press **`q`** inside the webcam video window to stop the program and release the camera.
