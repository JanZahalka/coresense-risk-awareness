"""
ml.featextr

Feature extraction module for risk awareness.
"""

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from riskam.ml import gaze, humandet

HUMAN_DETECTOR_MODEL = "hustvl/yolos-tiny"


def extract_human_risk_awareness_features(
    image: Image, visualize: bool = False
) -> dict:
    """
    Extract the features related to human risk from the given image.
    """

    # Extract human bounding boxes
    human_bboxes = humandet.detect_humans(image)

    # For each human, detect gaze
    human_gazes = [gaze.detect_gaze(image.crop(bbox)) for bbox in human_bboxes]

    # Return the extracted features
    features = {
        "human_bboxes": human_bboxes,
        "gaze": human_gazes,
    }

    # Visualization
    if visualize:
        draw = ImageDraw.Draw(image)

        for bbox, is_gazing in zip(human_bboxes, human_gazes):
            color = (
                "#00FF00" if is_gazing else "#FF0000"
            )  # Green if gazing, Red otherwise
            draw.rectangle(bbox, outline=color, width=3)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

        visualization = image
    else:
        visualization = None

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
