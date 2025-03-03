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
model = YOLO(YOLO_POSE_MODEL_PATH, verbose=False)


HEAD_OFFSET_THRESHOLD_RATIO = 0.25

FACE_OFFSET_LOWER_THRESHOLD_RATIO = 0.1
FACE_OFFSET_UPPER_THRESHOLD_RATIO = 0.1


def detect_humans(
    image_path: str,
) -> tuple[list[tuple[int, int, int, int]], list[float], list[bool]]:
    """
    Detects humans in the given image and whether they are likely aware of the robot,
    using YOLOv11n.
    """
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb, verbose=False)

    human_bboxes = []
    keypoints_np = None

    # Extract detections
    for result in results:

        # Append the bounding box
        if len(result.boxes.xyxy) == 0:
            continue  # Skip if no humans detected

        human_bboxes = result.boxes.xyxy.cpu().tolist()

        keypoints = result.keypoints

        # Extract only (x, y) coordinates
        keypoints_np = keypoints.data[..., :2].cpu().numpy()

    # Calculate bbox offsets from the center
    if len(image.shape) == 3:
        _, image_width, _ = image.shape  # Color image (3D)
    else:
        _, image_width = image.shape  # Grayscale image (2D)
    human_bbox_offsets = [
        _bbox_offset_score(bbox, image_width) for bbox in human_bboxes
    ]

    return human_bboxes, human_bbox_offsets, keypoints_np


def detect_gaze(keypoints_np: np.ndarray, human_idx: int) -> float:
    """
    Detects whether the person at the given index is looking at the robot.
    """
    # Zero keypoints detected -> no human looking at the robot
    if keypoints_np.shape[0] == 0:
        return 0.0

    kpts = keypoints_np[human_idx]

    if kpts.shape[0] == 0:  # Handle cases where keypoints exist but are empty
        # print(f"Skipping person {person_id + 1} due to missing keypoints")
        return 0.0

    # Extract keypoints
    nose = kpts[0]  # Nose (x, y)
    left_shoulder = kpts[5] if kpts.shape[0] > 5 else None
    right_shoulder = kpts[6] if kpts.shape[0] > 6 else None
    left_eye = kpts[1] if kpts.shape[0] > 1 else None
    right_eye = kpts[2] if kpts.shape[0] > 2 else None

    # # ====== METHOD 1: SHOULDER-BASED (Preferred if shoulders are visible) ======
    # if left_shoulder is not None and right_shoulder is not None:
    #     # Compute shoulder midpoint (acts as "neck" reference)
    #     neck_midpoint = (left_shoulder + right_shoulder) / 2
    #     shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

    #     # Compute head offset (nose vs. shoulders)
    #     head_offset = abs(nose[0] - neck_midpoint[0])
    #     normalized_offset = (
    #         head_offset / shoulder_width if shoulder_width > 0 else float("inf")
    #     )

    #     looking_at_robot = normalized_offset < HEAD_OFFSET_THRESHOLD_RATIO
    #     # print(
    #     #     f"Person {person_id+1} (Shoulder Method) Normalized Head Offset: {normalized_offset:.2f} | Looking: {looking_at_robot}"
    #     # )

    # ====== METHOD 2: FACE-BASED (Fallback if shoulders are occluded) ======
    if left_eye is not None and right_eye is not None:
        # Compute face width
        face_width = np.linalg.norm(right_eye - left_eye)

        # Use eye midpoint as the reference point instead of shoulders
        face_midpoint = (left_eye + right_eye) / 2
        face_offset = abs(nose[0] - face_midpoint[0])

        # Normalize by face width
        normalized_face_offset = (
            face_offset / face_width if face_width > 0 else float("inf")
        )

        looking_at_robot = min(
            1,
            max(
                0,
                (FACE_OFFSET_UPPER_THRESHOLD_RATIO - normalized_face_offset)
                / FACE_OFFSET_LOWER_THRESHOLD_RATIO,
            ),
        )
        # print(
        #     f"Person {person_id+1} (Face Method) Normalized Face Offset: {normalized_face_offset:.2f} | Looking: {looking_at_robot}"
        # )

    else:
        # No reliable keypoints available
        # print(f"Person {person_id+1} skipped due to missing keypoints.")
        looking_at_robot = 0.0

    return looking_at_robot


def _bbox_offset_score(bbox: tuple[int, int, int, int], image_width: int) -> float:
    """
    Calculates the offset score for the given bounding box.
    """
    x1, _, x2, _ = bbox
    bbox_center_x = (x1 + x2) / 2
    img_center_x = image_width / 2

    # Compute normalized offset from the motion axis
    offset = abs(bbox_center_x - img_center_x) / img_center_x

    # Compute motion-axis risk weighting (1 - offset^2)
    offset_score = 1 - offset**2

    return offset_score
