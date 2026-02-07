import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Model path
MODEL_PATH = "hand_landmarker.task"


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded.")


def process_video(video_path: str, output_path: str = None):
    """Process video with hand detection, finger bounding boxes, and blue edge highlighting."""

    download_model()

    # Create hand landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale, then back to BGR for colored overlays
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Convert to RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Get all landmark points
                h, w = frame.shape[:2]
                points = []
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))

                # Create hand mask for edge detection
                hand_mask = create_hand_mask(points, frame.shape)

                # Detect and draw hand edges in blue
                edges = cv2.Canny(cv2.bitwise_and(gray, gray, mask=hand_mask), 50, 150)
                output[edges > 0] = (255, 100, 0)  # Blue color (BGR)

                # Draw bounding boxes around each finger
                draw_finger_boxes(output, points)

        # Display the result
        cv2.imshow("Hand Detection", output)

        if writer:
            writer.write(output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Processing complete.")


def create_hand_mask(points: list, shape: tuple) -> np.ndarray:
    """Create a mask covering the hand region."""
    mask = np.zeros(shape[:2], dtype=np.uint8)

    if len(points) >= 21:
        # Create convex hull around hand landmarks
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)

        # Dilate to include edges
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_finger_boxes(frame: np.ndarray, points: list):
    """Draw bounding boxes around each finger."""
    if len(points) < 21:
        return

    # Define finger landmark ranges (base to tip)
    fingers = {
        "Thumb": [1, 2, 3, 4],
        "Index": [5, 6, 7, 8],
        "Middle": [9, 10, 11, 12],
        "Ring": [13, 14, 15, 16],
        "Pinky": [17, 18, 19, 20],
    }

    colors = [
        (0, 255, 0),  # Green
        (0, 255, 255),  # Yellow
        (0, 165, 255),  # Orange
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    for i, (finger_name, indices) in enumerate(fingers.items()):
        finger_points = [points[idx] for idx in indices]

        # Calculate bounding box
        x_coords = [p[0] for p in finger_points]
        y_coords = [p[1] for p in finger_points]

        padding = 10
        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(frame.shape[1], max(x_coords) + padding)
        y_max = min(frame.shape[0], max(y_coords) + padding)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colors[i], 2)

        # Add label
        cv2.putText(
            frame,
            finger_name,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colors[i],
            1,
        )


if __name__ == "__main__":
    # Change this to your video file path
    VIDEO_PATH = r"fingers.mp4"
    OUTPUT_PATH = "fingers_edited.mp4"  # Set to None to skip saving

    process_video(VIDEO_PATH, OUTPUT_PATH)
