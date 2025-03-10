"""
riskam.visualization

Visualization tools for the risk awareness module.
"""

from pathlib import Path

import cv2
import numpy as np

from riskam.score import RISK_SCORE_BREAKPOINTS

# pylint: disable=no-member


def visualize_risk(
    image_path: str,
    output_path: str | None,
    bboxes: list[tuple[int, int, int, int]],
    rel_depth: np.ndarray,
    risk_features: dict[str, np.ndarray],
    risk_score: float,
    max_risk_idx: int,
):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Add rel_depth overlay: closest pixels remain as-is and further ones fade to dark gray
    # Normalize rel_depth to range [0,1]
    norm_depth = (rel_depth - rel_depth.min()) / (
        rel_depth.max() - rel_depth.min() + 1e-6
    )
    # Create a dark gray overlay image
    overlay = np.full_like(image, (50, 50, 50))
    # Soften overlay: reduce effect of depth on blending (e.g., only half as strong)
    soft_weight = norm_depth * 0.85
    # Blend each pixel with the softer weight
    image = (
        image.astype(np.float32) * (1 - soft_weight[..., None])
        + overlay.astype(np.float32) * soft_weight[..., None]
    )
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Draw bounding boxes with color based on gaze score
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gaze_score = risk_features["gaze"][i]
        if gaze_score == 0:
            color = (0, 0, 255)  # red
        elif gaze_score == 1:
            color = (0, 255, 0)  # green
        else:
            color = (0, 255, 255)  # yellow
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add risk score overlay in the top right corner
    text = f"{risk_score:.3f}"
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 600)
    thickness = int(font_scale * 2)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = width - text_size[0] - 20
    text_y = 40
    # Background rectangle for text
    cv2.rectangle(
        image,
        (text_x - 10, text_y - 30),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )
    # Determine text color based on RISK_SCORE_BREAKPOINTS
    if risk_score == RISK_SCORE_BREAKPOINTS[0]:
        score_color = (255, 200, 150)  # light blue
    elif risk_score <= RISK_SCORE_BREAKPOINTS[1]:
        score_color = (0, 255, 0)  # green
    elif risk_score <= RISK_SCORE_BREAKPOINTS[2]:
        score_color = (0, 255, 255)  # yellow
    else:
        score_color = (0, 0, 255)  # red
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        score_color,
        thickness,
        cv2.LINE_AA,
    )

    # Mark the bounding box corresponding to max_risk_idx with an asterisk centered in the box
    if 0 <= max_risk_idx < len(bboxes):
        x1, y1, x2, y2 = bboxes[max_risk_idx]
        # Convert coordinates to int and compute center of bounding box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # Compute text size for the asterisk
        star = "*"
        star_size, baseline = cv2.getTextSize(star, font, font_scale, thickness)
        # Adjust position so that the center of the asterisk text is at 'center'
        star_x = center[0] - star_size[0] // 2
        star_y = center[1] + star_size[1] // 2
        # Draw asterisk with an outline for visibility
        cv2.putText(
            image,
            star,
            (star_x, star_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            star,
            (star_x, star_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if output_path is None:
        # Show on screen
        cv2.imshow("Risk Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Ensure directory exists and save image
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
