"""
riskam.score

The risk awareness score computation module.
"""

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# pylint: disable=no-member


def risk_awareness_score(
    features: dict,
    w_proximity: float = 0.65,
    w_gaze: float = 0.3,
    w_position: float = 0.05,
) -> float:
    """
    For the given image, calculates the risk awareness score.
    """

    # If no features are available, return 0 (there are no humans)
    if features is None:
        return 0.0

    # Calculate the motion risk
    # motion_risk = 0.5 * min(1, features["proximity_niqr"])

    # Gaze penalty (reduce risk if gaze is True)
    gaze_risk = w_gaze * features["gaze"]

    # Proximity risk with quadratic scaling and x-offset taken into account
    proximity_risk = w_proximity * (1 - features["proximity"])
    position_risk = w_position * features["x_offset_score"]

    # Calculate risk score
    risk_score = proximity_risk + gaze_risk + position_risk

    # Clamp to [0.00001, 1] and return
    return max(0.00001, min(1, risk_score))


def risk_score_image_overlay(pil_image: Image, risk_score: float) -> Image:
    # Convert PIL Image to OpenCV BGR format
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Define risk categories and colors (BGR)
    if risk_score == 0:
        color = (255, 200, 150)  # Light blue
    elif risk_score <= 0.3:
        color = (0, 255, 0)  # Green
    elif risk_score <= 0.6:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    text = f"{risk_score:.3f}"

    height, width = image_cv.shape[:2]

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 600)
    thickness = int(font_scale * 2)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = width - text_size[0] - 20
    text_y = 40

    rect_x1, rect_y1 = text_x - 10, text_y - 30
    rect_x2, rect_y2 = text_x + text_size[0] + 10, text_y + 10
    cv2.rectangle(image_cv, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

    cv2.putText(
        image_cv,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # Convert image back to PIL.Image
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # ...existing code for displaying image...
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_rgb)
    # plt.axis("off")
    # plt.show()

    return Image.fromarray(image_rgb)
