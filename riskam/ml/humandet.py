"""
ml.humandet

Human detection for risk awareness.
"""

from PIL import Image
import torch
from transformers import (
    YolosImageProcessor,
    YolosForObjectDetection,
)

OBJECT_DETECTION_MODEL = "hustvl/yolos-tiny"

processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")


def detect_humans(image: Image) -> list:
    """
    Detects humans in the given image. Returns a list of bounding boxes.
    """
    # Use a pre-trained model to detect humans in the image
    # Return a list of bounding boxes for the detected humans

    # Process the image
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(  # pylint: disable=no-member
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]

    bboxes = []

    for label, box in zip(results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] != "person":
            continue

        bboxes.append(box.tolist())

    return bboxes
