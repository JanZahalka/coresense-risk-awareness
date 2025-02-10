"""
ml.featextr

Feature extraction module for risk awareness.
"""

from time import time

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from riskam.ml import depth, humandet

HUMAN_DETECTOR_MODEL = "hustvl/yolos-tiny"


def extract_human_risk_awareness_features(
    image: Image, image_path: str, visualize: bool = False
) -> dict:
    """
    Extract the features related to human risk from the given image.
    """
    t = time()
    human_bboxes, human_gazes = humandet.detect_humans(image_path)

    # Return the extracted features
    features = {
        "human_bboxes": human_bboxes,
        "gaze": human_gazes,
    }

    rel_depth = depth.estimate_depth(image_path)
    image = Image.open(image_path).convert("RGBA")

    # Normalize depth values between 0 (far) and 1 (close)
    depth_min, depth_max = rel_depth.min(), rel_depth.max()
    rel_depth_normalized = 1 - (rel_depth - depth_min) / (
        depth_max - depth_min
    )  # Invert depth (close=1, far=0)

    alpha_channel = (rel_depth_normalized * 255).astype(
        np.uint8
    )  # Alpha (transparency) based on depth
    black_channel = np.zeros_like(alpha_channel)  # Black color (0,0,0)

    # Stack into RGBA image (depth overlay)
    depth_overlay = np.stack(
        [black_channel, black_channel, black_channel, alpha_channel], axis=-1
    )

    # Convert to PIL image and resize to match input image
    depth_overlay_pil = Image.fromarray(depth_overlay, mode="RGBA").resize(image.size)

    # Blend images (alpha composite)
    overlayed_image = Image.alpha_composite(image, depth_overlay_pil)

    # Visualization
    if visualize:
        draw = ImageDraw.Draw(overlayed_image)

        for bbox, is_gazing in zip(human_bboxes, human_gazes):
            color = (
                "#00FF00" if is_gazing else "#FF0000"
            )  # Green if gazing, Red otherwise
            draw.rectangle(bbox, outline=color, width=3)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(overlayed_image)
        plt.axis("off")
        plt.show()

        visualization = overlayed_image
    else:
        visualization = None

    print("Feature extraction time:", time() - t)

    return features, visualization

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
