"""
ml.depth

The depth estimation module (based on MiDaS).

Using https://pytorch.org/hub/intelisl_midas_v2/
"""

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt


# OpenCV throws no-member linting errors
# pylint: disable=no-member


MIDAS_MODEL_TYPE = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform


def estimate_depth(
    image_path: str, human_bboxes: list[list[int]]
) -> tuple[np.ndarray, dict]:
    """
    Estimates depth for the given image.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_batch = transform(image).to(device)

    # Predict
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    inverse_depths = prediction.cpu().numpy()

    # Normalize, use log normalization to match human depth perception
    # (going from 1 meter to 2 is way more significant than from 10 to 11)
    inv_depths_min = inverse_depths.min()
    inv_depths_max = inverse_depths.max()

    # Apply the correct inverse log normalization
    normalized_inverse_depths = (
        np.log(inv_depths_max + 1) - np.log(inverse_depths + 1)
    ) / (np.log(inv_depths_max + 1) - np.log(inv_depths_min + 1))

    # Calculate the stats
    depth_stats = _closest_human_depth_stats(normalized_inverse_depths, human_bboxes)

    return normalized_inverse_depths, depth_stats


def _closest_human_depth_stats(
    inverse_depths: np.ndarray, human_bboxes: list[list[int]]
) -> dict:
    """
    Returns the depth stats corresponding to the CLOSEST human in the image.
    """

    # Image-wide stats
    inv_depths_std = np.std(inverse_depths)

    # Bounding box medians
    bbox_inv_depths_all = []
    bbox_p10s = []

    # Iterate over the bounding boxes
    for bbox in human_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Slice the image
        bbox_inv_depths = inverse_depths[y1:y2, x1:x2]
        bbox_inv_depths_all.append(bbox_inv_depths)

        # Calculate the median & append
        bbox_p10s.append(np.percentile(bbox_inv_depths, 10))

    # Select the closest human
    closest_idx = np.argmin(bbox_p10s)

    # Calculate the closest human's normalized IQR - if there is no variation, set to 0
    p25, p75 = np.percentile(bbox_inv_depths_all[closest_idx], [25, 75])

    if inv_depths_std == 0.0:
        closest_niqr = 0.0
    else:
        closest_niqr = (p75 - p25) / inv_depths_std

    return {
        "closest_human_idx": closest_idx,
        "closest_human_p10": bbox_p10s[closest_idx],
        "closest_human_niqr": closest_niqr,
    }
