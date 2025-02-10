"""
ml.humandet

Human detection for risk awareness.
"""

from pathlib import Path

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

from riskam.data.paths import ML_MODELS_DIR

# OpenCV throws no-member linting errors
# pylint: disable=no-member

# Check if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO Pose model on GPU
YOLO_POSE_MODEL_PATH = ML_MODELS_DIR / "yolo11n-pose.pt"
model = YOLO(YOLO_POSE_MODEL_PATH)


HEAD_OFFSET_THRESHOLD_RATIO = 0.25
FACE_OFFSET_THRESHOLD_RATIO = 0.3


def detect_humans(
    image_path: str,
) -> tuple[list[tuple[int, int, int, int]], list[bool]]:
    """
    Detects humans in the given image and whether they are likely aware of the robot,
    using YOLOv11n.
    """
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)

    human_bboxes = []
    human_gazes = []

    # Extract detections
    for result in results:
        # Append the bounding box

        if len(result.boxes.xyxy) == 0:
            continue  # Skip if no humans detected

        human_bboxes = result.boxes.xyxy.cpu().tolist()

        keypoints = result.keypoints
        # Extract only (x, y) coordinates

        keypoints_np = keypoints.data[..., :2].cpu().numpy()

        # Zero keypoints detected
        if keypoints_np.shape[0] == 0:
            continue

        # Iterate over detected humans
        for person_id, kpts in enumerate(keypoints_np):
            if kpts.shape[0] == 0:  # Handle cases where keypoints exist but are empty
                print(f"Skipping person {person_id + 1} due to missing keypoints")
                continue

            # Extract keypoints
            nose = kpts[0]  # Nose (x, y)
            left_shoulder = kpts[5] if kpts.shape[0] > 5 else None
            right_shoulder = kpts[6] if kpts.shape[0] > 6 else None
            left_eye = kpts[1] if kpts.shape[0] > 1 else None
            right_eye = kpts[2] if kpts.shape[0] > 2 else None

            # ====== METHOD 1: SHOULDER-BASED (Preferred if shoulders are visible) ======
            if left_shoulder is not None and right_shoulder is not None:
                # Compute shoulder midpoint (acts as "neck" reference)
                neck_midpoint = (left_shoulder + right_shoulder) / 2
                shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

                # Compute head offset (nose vs. shoulders)
                head_offset = abs(nose[0] - neck_midpoint[0])
                normalized_offset = (
                    head_offset / shoulder_width if shoulder_width > 0 else float("inf")
                )

                looking_at_robot = normalized_offset < HEAD_OFFSET_THRESHOLD_RATIO
                print(
                    f"Person {person_id+1} (Shoulder Method) Normalized Head Offset: {normalized_offset:.2f} | Looking: {looking_at_robot}"
                )

            # ====== METHOD 2: FACE-BASED (Fallback if shoulders are occluded) ======
            elif left_eye is not None and right_eye is not None:
                # Compute face width
                face_width = np.linalg.norm(right_eye - left_eye)

                # Use eye midpoint as the reference point instead of shoulders
                face_midpoint = (left_eye + right_eye) / 2
                face_offset = abs(nose[0] - face_midpoint[0])

                # Normalize by face width
                normalized_face_offset = (
                    face_offset / face_width if face_width > 0 else float("inf")
                )

                looking_at_robot = normalized_face_offset < FACE_OFFSET_THRESHOLD_RATIO
                print(
                    f"Person {person_id+1} (Face Method) Normalized Face Offset: {normalized_face_offset:.2f} | Looking: {looking_at_robot}"
                )

            else:
                # No reliable keypoints available
                print(f"Person {person_id+1} skipped due to missing keypoints.")
                looking_at_robot = False

            human_gazes.append(looking_at_robot)

    return human_bboxes, human_gazes
