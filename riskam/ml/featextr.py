"""
ml.featextr

Feature extraction module for risk awareness.
"""

from time import time


from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from riskam.ml import depth, humandet
from riskam import score

HUMAN_DETECTOR_MODEL = "hustvl/yolos-tiny"


def extract_human_risk_awareness_features(
    image: Image, image_path: str, visualize: bool = False
) -> dict:
    """
    Extract the features related to human risk from the given image.
    """
    t = time()

    # Extract human bounding boxes, offset scores, and keypoints
    human_bboxes, offset_scores, keypoints_np = humandet.detect_humans(image_path)

    # Estimate depth (MiDaS)
    rel_depth, depth_features = depth.estimate_depth(image_path, human_bboxes)

    # Process the features
    if len(human_bboxes) > 0:
        ch = depth_features["closest_human_idx"]

        # Extract the human's gaze
        gaze_score = humandet.detect_gaze(keypoints_np, ch)

        risk_features = {
            "proximity": depth_features["closest_human_p10"],
            "x_offset_score": offset_scores[ch],
            "gaze": gaze_score,
        }
    # If there are no humans, return None
    else:
        risk_features = None

    time_elapsed = time() - t

    if visualize:
        image = Image.open(image_path).convert("RGBA")

        alpha_channel = (rel_depth * 255).astype(
            np.uint8
        )  # Alpha (transparency) based on depth
        dark_gray = np.full_like(alpha_channel, 64)  # Dark gray

        # Stack into RGBA image (depth overlay)
        depth_overlay = np.stack(
            [dark_gray, dark_gray, dark_gray, alpha_channel], axis=-1
        )

        # Convert to PIL image and resize to match input image
        depth_overlay_pil = Image.fromarray(depth_overlay, mode="RGBA").resize(
            image.size
        )

        # Blend images (alpha composite)
        overlayed_image = Image.alpha_composite(image, depth_overlay_pil)

        draw = ImageDraw.Draw(overlayed_image)

        if len(human_bboxes) > 0:
            bbox = human_bboxes[ch]

            if gaze_score == 1.0:
                color = "#00FF00"
            elif gaze_score == 0.0:
                color = "#FF0000"
            else:
                color = "#FFFF00"
            draw.rectangle(bbox, outline=color, width=3)

        # # Display the image
        # plt.figure(figsize=(8, 8))
        # plt.imshow(overlayed_image)
        # plt.axis("off")
        # plt.show()

        visualization = overlayed_image
    else:
        visualization = None

    return risk_features, visualization

    # Estimate the distance of the closest human
    # - MiDaS: https://github.com/isl-org/MiDaS
    # - Works great!
    # Estimate the human's emotion (discomfort)
    # - FER (facial expression recognition)
    # - AffectNet looks good -> although this is JUST on faces -> need to cut
    # Estimate the human's gaze (is he looking at the robot?)
    # Estimate environmental hazards


def _detect_humans(image: Image) -> list:
    """
    Detect humans in the given image. Returns a list of bounding boxes.
    """

    # Use a pre-trained model to detect humans in the image
    # Return a list of bounding boxes for the detected humans
