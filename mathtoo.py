import cv2
import mediapipe as mp
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging
import time
import threading

logging.getLogger('tensorflow').setLevel(logging.ERROR)


class GestureCaptioningApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.drawn_points = []
        self.is_drawing = False
        self.shape_completed = False
        self.info_displayed = ""
        self.last_caption_time = 0
        self.caption_cooldown = 3
        self.latest_caption = ""
        self.processing_caption = False

        self.modes = ["Draw & Caption", "Object Recognition", "Text Translation", "Document Analysis"]
        self.current_mode = 0

        self.model_ready = False
        threading.Thread(target=self.load_model).start()

    def load_model(self):
        """Load the captioning model in a background thread"""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model_ready = True
        print("Model loaded successfully")

    def get_hand_gesture(self, frame):
        """Detect hand gestures and return fingertip position and gesture type"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        fingertip, gesture = None, "none"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get fingertip (index finger tip)
                fingertip = (
                    int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                    int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                )

                # Detect pointing gesture (index finger extended, others closed)
                if (hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y):

                    # Check if other fingers are closed
                    other_fingers_closed = True
                    for finger_tip, finger_pip in [
                        (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                        (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
                        (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
                    ]:
                        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y:
                            other_fingers_closed = False
                            break

                    if other_fingers_closed:
                        gesture = "pointing"

                # Detect pinch gesture for capture
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate distance between thumb and index finger
                distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                if distance < 0.05:  # Threshold for pinch detection
                    gesture = "pinch"

                # Draw fingertip for visualization
                cv2.circle(frame, fingertip, 10, (0, 255, 0), -1)

        return fingertip, gesture

    def get_caption(self, frame, drawn_points):
        """Generate caption for the area inside the drawn shape"""
        if not self.model_ready or len(drawn_points) < 5:
            return "Loading model or not enough points..."

        try:
            # Create a mask for the area inside the drawn shape
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            points_array = np.array(drawn_points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)

            # Apply mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Get bounding box
            x1, y1 = max(0, min([p[0] for p in drawn_points])), max(0, min([p[1] for p in drawn_points]))
            x2, y2 = min(frame.shape[1], max([p[0] for p in drawn_points])), min(frame.shape[0],
                                                                                 max([p[1] for p in drawn_points]))

            # Ensure the region is not empty
            if x2 <= x1 or y2 <= y1 or x2 - x1 < 10 or y2 - y1 < 10:
                return "Selected area too small"

            # Extract only the region inside the drawn shape
            roi = masked_frame[y1:y2, x1:x2]

            # Convert to RGB for the model
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)

            # Process with captioning model
            inputs = self.processor(images=roi_pil, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=30)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Customize caption based on mode
            if self.current_mode == 1:  # Object Recognition
                caption = "Object: " + caption
            elif self.current_mode == 2:  # Text Translation
                # Placeholder for OCR and translation
                caption = "Translation: " + caption
            elif self.current_mode == 3:  # Document Analysis
                caption = "Document: " + caption

            return caption
        except Exception as e:
            return f"Error: {str(e)}"

    def draw_help_overlay(self, frame):
        """Draw help text with keyboard shortcuts"""
        help_text = [
            "KEYBOARD CONTROLS:",
            "DEL: Clear drawing",
            "SPACE: Capture/Generate caption",
            "M: Change mode",
            "ESC or Q: Quit"
        ]

        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 250, 10), (frame.shape[1] - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Add text
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (frame.shape[1] - 240, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally
            fingertip, gesture = self.get_hand_gesture(frame)

            # Handle different gestures
            if fingertip:
                x, y = fingertip

                # Start drawing with pointing gesture
                if gesture == "pointing" and not self.shape_completed:
                    if not self.is_drawing:
                        self.drawn_points = [(x, y)]
                        self.is_drawing = True
                    else:
                        self.drawn_points.append((x, y))

                # Complete shape with pinch gesture
                if gesture == "pinch" and self.is_drawing and len(self.drawn_points) > 5:
                    self.shape_completed = True
                    self.is_drawing = False

                    # Process caption
                    if not self.processing_caption and time.time() - self.last_caption_time > self.caption_cooldown:
                        self.process_caption(frame.copy())

            # Draw all points in the list
            if len(self.drawn_points) > 1:
                points_array = np.array(self.drawn_points, dtype=np.int32)
                cv2.polylines(frame, [points_array], self.shape_completed, (255, 0, 0), 2)

                if self.shape_completed:
                    # Fill the shape with semi-transparent color
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points_array], (0, 150, 0, 128))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Display mode information
            cv2.putText(frame, f"Mode: {self.modes[self.current_mode]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display caption
            if self.latest_caption:
                # Create a semi-transparent background for text
                text_size = cv2.getTextSize(self.latest_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x, text_y = 10, 70
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0, 128), -1)
                cv2.putText(frame, self.latest_caption, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display processing indicator
            if self.processing_caption:
                cv2.putText(frame, "Processing...", (frame.shape[1] - 150, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Add help overlay
            self.draw_help_overlay(frame)

            # Display the frame
            cv2.imshow("Smart Captioning System", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            elif key == ord('m'):  # 'm' to change mode
                self.current_mode = (self.current_mode + 1) % len(self.modes)
                self.info_displayed = f"Mode: {self.modes[self.current_mode]}"
            elif key == 32:  # SPACE to capture/generate caption
                if len(self.drawn_points) > 5:
                    self.shape_completed = True
                    self.is_drawing = False
                    self.process_caption(frame.copy())
            elif key == 8 or key == 127:  # BACKSPACE or DEL to clear
                self.drawn_points = []
                self.is_drawing = False
                self.shape_completed = False
                self.latest_caption = ""

        cap.release()
        cv2.destroyAllWindows()

    def process_caption(self, frame):
        """Process the caption in a separate thread"""
        if self.processing_caption or time.time() - self.last_caption_time <= self.caption_cooldown:
            return

        self.processing_caption = True
        self.last_caption_time = time.time()

        def generate_caption():
            self.latest_caption = self.get_caption(frame, self.drawn_points)
            self.processing_caption = False

        threading.Thread(target=generate_caption).start()


if __name__ == "__main__":
    app = GestureCaptioningApp()
    app.run()