"""
Finger Counter - i2i Academy Computer Vision Assignment
This script implements a real-time finger-counting application using OpenCV
and MediaPipe, structured with software design patterns (Strategy, Factory,
Facade, and Composition).
"""

"""
Requirements:
    pip install opencv-python mediapipe

Run:
    python hand_tracking.py

Press 'q' to quit the application.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import time
import cv2
import mediapipe as mp


# ==========================================
# 1. Strategy Pattern: Finger Base & Subclasses
# ==========================================

class Finger(ABC):
    """
    Abstract base class representing a general finger interface.
    Each specific finger algorithm inherits from this class.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_open(self, landmarks) -> bool:
        """
        Determines whether the finger is open based on the landmark coordinates.
        """
        pass


class StandardFinger(Finger):
    """
    Concrete strategy for standard fingers (Index, Middle, Ring, Pinky).
    Uses relative y-coordinate comparison between the Tip and PIP joints.
    """
    def __init__(self, name: str, tip_idx: int, pip_idx: int):
        super().__init__(name)
        self.tip_idx = tip_idx
        self.pip_idx = pip_idx

    def is_open(self, landmarks) -> bool:
        # y increases downwards in screen coordinates.
        # Tip y < PIP y means the tip is higher on screen (finger is extended/open).
        return landmarks[self.tip_idx].y < landmarks[self.pip_idx].y


class Thumb(Finger):
    """
    Concrete strategy for the Thumb.
    Uses horizontal x-coordinate checks adjusted dynamically by hand orientation.
    """
    def __init__(self, name: str, tip_idx: int, ip_idx: int, mcp_idx: int, pinky_mcp_idx: int):
        super().__init__(name)
        self.tip_idx = tip_idx
        self.ip_idx = ip_idx
        self.mcp_idx = mcp_idx
        self.pinky_mcp_idx = pinky_mcp_idx

    def is_open(self, landmarks) -> bool:
        thumb_tip = landmarks[self.tip_idx]
        thumb_ip = landmarks[self.ip_idx]
        thumb_mcp = landmarks[self.mcp_idx]
        pinky_mcp = landmarks[self.pinky_mcp_idx]

        # Determine hand orientation (whether thumb is on the left or right side)
        # by comparing Pinky MCP x-coordinate with Thumb MCP x-coordinate.
        if pinky_mcp.x > thumb_mcp.x:
            # Thumb is on the left side of the hand relative to the camera view
            # (e.g., Right hand palm facing the camera).
            return thumb_tip.x < thumb_ip.x
        else:
            # Thumb is on the right side of the hand relative to the camera view
            # (e.g., Left hand palm facing the camera).
            return thumb_tip.x > thumb_ip.x


# ==========================================
# 2. Factory Pattern: Finger Instantiation
# ==========================================

class FingerFactory:
    """
    Factory class to instantiate finger strategy objects with correct landmarks.
    """
    @staticmethod
    def create_finger(name: str) -> Finger:
        # Landmark indices reference the 21 MediaPipe hand points:
        # - Thumb: Tip (4), IP (3), MCP (2), Pinky MCP (17)
        # - Index: Tip (8), PIP (6)
        # - Middle: Tip (12), PIP (10)
        # - Ring: Tip (16), PIP (14)
        # - Pinky: Tip (20), PIP (18)
        if name == "Thumb":
            return Thumb("Thumb", tip_idx=4, ip_idx=3, mcp_idx=2, pinky_mcp_idx=17)
        elif name == "Index":
            return StandardFinger("Index", tip_idx=8, pip_idx=6)
        elif name == "Middle":
            return StandardFinger("Middle", tip_idx=12, pip_idx=10)
        elif name == "Ring":
            return StandardFinger("Ring", tip_idx=16, pip_idx=14)
        elif name == "Pinky":
            return StandardFinger("Pinky", tip_idx=20, pip_idx=18)
        else:
            raise ValueError(f"Unknown finger name: {name}")


# ==========================================
# 3. Composition: Hand representation
# ==========================================

class Hand:
    """
    Represents a hand composed of different fingers.
    Calculates overall state based on individual finger evaluation strategies.
    """
    def __init__(self):
        # Compose the hand of five Finger strategy objects created via factory
        self.fingers: List[Finger] = [
            FingerFactory.create_finger("Thumb"),
            FingerFactory.create_finger("Index"),
            FingerFactory.create_finger("Middle"),
            FingerFactory.create_finger("Ring"),
            FingerFactory.create_finger("Pinky")
        ]
        self.finger_states: Dict[str, bool] = {}

    def update_states(self, landmarks) -> Tuple[int, Dict[str, bool]]:
        """
        Updates the state (open/closed) of each finger using their respective logic.
        Returns the total open finger count and dictionary of individual states.
        """
        self.finger_states = {
            finger.name: finger.is_open(landmarks) for finger in self.fingers
        }
        total_open = sum(1 for state in self.finger_states.values() if state)
        return total_open, self.finger_states


# ==========================================
# 4. Facade Pattern: MediaPipe Hand Tracker
# ==========================================

class HandTracker:
    """
    Facade class that wraps the MediaPipe Hands module to simplify usage.
    """
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame_rgb):
        """Processes the RGB frame and returns the detection results."""
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws hand skeletal landmarks and connections on the BGR frame."""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(46, 204, 113), thickness=2, circle_radius=3),  # Green landmarks
            self.mp_draw.DrawingSpec(color=(236, 240, 241), thickness=2)                 # White connections
        )


# ==========================================
# 5. Application Controller / Orchestrator
# ==========================================

class FingerCounterApp:
    """
    Main application orchestrator that handles camera streaming,
    coordinate processing, and visual HUD drawing.
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.tracker = HandTracker()
        self.hand_model = Hand()
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam at index {self.camera_index}. Please check connection.")
            return

        print("\n" + "="*50)
        print("i2i Academy - Computer Vision Finger Counter (OOP Refactored) started!")
        print("Press 'q' in the video window to quit.")
        print("="*50 + "\n")

        prev_time = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Error: Failed to read frame from webcam.")
                break

            # Flip frame horizontally for a more natural mirror perspective
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Convert BGR frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker.process_frame(rgb_frame)

            total_fingers = 0
            hand_states = {}

            if results.multi_hand_landmarks:
                detected_wrists = []
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    # Check if this wrist is too close to any already processed wrist
                    # (Distance threshold of 0.08 normalized coordinates is about 8% of screen)
                    is_duplicate = False
                    for prev_wrist in detected_wrists:
                        dist = ((wrist.x - prev_wrist.x)**2 + (wrist.y - prev_wrist.y)**2)**0.5
                        if dist < 0.08:
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue  # Skip duplicate tracking of the same hand
                    
                    detected_wrists.append(wrist)

                    # Draw skeletal outline on the display frame
                    self.tracker.draw_landmarks(frame, hand_landmarks)

                    # Update and fetch state calculations for this hand
                    count, states = self.hand_model.update_states(hand_landmarks.landmark)
                    total_fingers += count
                    
                    # Accumulate states to display (showing latest evaluated hand status)
                    hand_states.update(states)

                # Draw individual finger status labels
                y_offset = 120
                for finger_name, is_open in hand_states.items():
                    color = (46, 204, 113) if is_open else (231, 76, 60)  # Green for Open, Red for Closed
                    status_text = f"{finger_name}: {'OPEN' if is_open else 'CLOSED'}"
                    cv2.putText(frame, status_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25

            # Calculate Frame Rate (FPS)
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
            prev_time = curr_time

            # Draw Modern HUD / Overlay Panels
            # 1. Dark semi-transparent panel for total count display
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 95), (44, 62, 80), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 2. Total finger count text
            cv2.putText(
                frame,
                f"Fingers: {total_fingers}",
                (20, 65),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (241, 196, 15),  # Gold color
                3
            )

            # 3. FPS display
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (52, 152, 219),  # Light blue color
                2
            )

            # Show the final frame
            cv2.imshow("i2i Academy - Hand Tracking & Finger Counter", frame)

            # Key check to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release webcam resources and close window
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully.")


if __name__ == "__main__":
    app = FingerCounterApp()
    app.run()
