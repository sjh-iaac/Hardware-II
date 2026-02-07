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


def is_finger_extended(points: list, finger_indices: list, is_thumb: bool = False) -> bool:
    """Check if a finger is extended based on landmark positions."""
    if len(points) < 21:
        return False

    tip = points[finger_indices[3]]  # Fingertip
    pip = points[finger_indices[2]]  # Middle joint
    mcp = points[finger_indices[0]]  # Base joint (knuckle)

    if is_thumb:
        # For thumb, check horizontal distance from palm
        palm_center = points[0]  # Wrist
        return abs(tip[0] - palm_center[0]) > abs(mcp[0] - palm_center[0])
    else:
        # For other fingers, tip should be above (lower y) the PIP joint
        return tip[1] < pip[1]


def process_video(video_path: str, output_path: str = None, darkness: float = 0.5):
    """Process video with extended finger detection and darker grayscale."""

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

        # Convert to darker grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = (gray * darkness).astype(np.uint8)  # Darken the grayscale
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

                # Draw bounding boxes around extended fingers only
                draw_extended_finger_boxes(output, points)

        # Display the result
        cv2.imshow("Hand Detection v2", output)

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


def draw_extended_finger_boxes(frame: np.ndarray, points: list):
    """Draw bounding boxes around extended fingers only with index counter."""
    if len(points) < 21:
        return

    # Define finger landmark ranges (base to tip)
    fingers = [
        ([1, 2, 3, 4], True),   # Thumb (is_thumb=True)
        ([5, 6, 7, 8], False),  # Index
        ([9, 10, 11, 12], False),  # Middle
        ([13, 14, 15, 16], False),  # Ring
        ([17, 18, 19, 20], False),  # Pinky
    ]

    color = (0, 255, 0)  # Green for all boxes
    finger_counter = 1

    for indices, is_thumb in fingers:
        # Only draw box if finger is extended
        if not is_finger_extended(points, indices, is_thumb):
            continue

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
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Add index label
        cv2.putText(
            frame,
            str(finger_counter),
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        finger_counter += 1


if __name__ == "__main__":
    # Change this to your video file path
    VIDEO_PATH = r"fingers.mp4"
    OUTPUT_PATH = "fingers_edited_v2.mp4"  # Set to None to skip saving

    # darkness: 0.0 = black, 1.0 = normal, lower = darker
    process_video(VIDEO_PATH, OUTPUT_PATH, darkness=0.4)
