"""
riskam.score

The risk awareness score computation module.
"""

import cv2
import matplotlib.pyplot as plt
from PIL import Image

# pylint: disable=no-member


def risk_awareness_score(features: dict) -> float:
    """
    For the given image, calculates the risk awareness score.
    """

    # If no features are available, return 0 (there are no humans)
    if features is None:
        return 0.0

    # Calculate the motion risk
    # motion_risk = 0.5 * min(1, features["proximity_niqr"])

    # Gaze penalty (reduce risk if gaze is True)
    gaze_penalty = -0.3 if features["gaze"] else 0.0

    # Proximity risk with quadratic scaling and x-offset taken into account
    proximity_risk = (1 - features["proximity"]) ** 1.5
    position_weight = (1 - features["x_offset_score"]) ** 1.5

    # Calculate risk score
    risk_score = proximity_risk + gaze_penalty + position_weight

    # Clamp to [0, 1] and return
    return max(0, min(1, risk_score))


def risk_score_image_overlay(image_path: str, risk_score: float) -> Image:
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Define risk categories and colors (BGR format for OpenCV)
    if risk_score == 0:
        risk_text = "No Risk"
        color = (255, 200, 150)  # Light blue
    elif risk_score <= 0.3:
        risk_text = "Low Risk"
        color = (0, 255, 0)  # Green
    elif risk_score <= 0.6:
        risk_text = "Medium Risk"
        color = (0, 255, 255)  # Yellow
    else:
        risk_text = "High Risk"
        color = (0, 0, 255)  # Red

    # Format risk text with score
    text = f"{risk_text} ({risk_score:.2f})"

    # Get image dimensions
    height, width = image.shape[:2]

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 600)  # Scale font size based on image width
    thickness = int(font_scale * 2)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Define text position (upper-right corner, slightly padded)
    text_x = width - text_size[0] - 20
    text_y = 40

    # Draw background rectangle for contrast
    rect_x1, rect_y1 = text_x - 10, text_y - 30
    rect_x2, rect_y2 = text_x + text_size[0] + 10, text_y + 10
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

    # Overlay text
    cv2.putText(
        image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA
    )

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

    return Image.fromarray(image_rgb)
