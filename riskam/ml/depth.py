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


def estimate_depth(image_path: str) -> np.ndarray:
    """
    Estimates depth for the given image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    return output
