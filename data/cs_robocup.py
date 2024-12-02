"""
data.cs_robocup

The CoreSense RoboCup dataset recorded with a social robot.

https://zenodo.org/records/13748065
"""

import argparse
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import YolosImageProcessor, YolosForObjectDetection
from tqdm import tqdm

from paths import CS_ROBOCUP_ML_RAW_DIR


# Script constants
ROUTINES = ["detect_humans"]


class CSRoboCup(Dataset):
    """
    The CoreSense RoboCup PyTorch dataset object.

    Currently implemented as a SUPERVISED dataset with run identifiers as labels. That said,
    it's an unsupervised dataset for ML intents and purposes.
    """

    def __init__(self, root_dir: str, transform=None):
        # Initialize the dataset's variables
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Fill the dataset
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name / "rgb"
            for img_path in cls_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def detect_humans_in_frames() -> None:
    """
    Go over the CoreSense Robocup dataset and for each frame, extract the number of humans in it
    and the corresponding bounding boxes.
    """

    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    # Iterate over the runs
    for rb in range(1, 9):
        print("+++ PROCESSING RUN RB_0{rb} +++")

        rb_rgb_dir = CS_ROBOCUP_ML_RAW_DIR / f"RB_0{rb}" / "rgb"

        for img_path in tqdm(rb_rgb_dir.glob("*.png")):
            # Load the image
            image = Image.open(img_path).convert("RGB")

            # Process the image
            inputs = processor(image, return_tensors="pt")
            outputs = model(**inputs)

            # Logits and bounding boxes
            logits = outputs.logits
            bboxes = outputs.pred_boxes

            # Process the results
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if model.config.id2label[label.item()] != "person":
                    continue

                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

            break

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS RoboCup dataset service.")
    parser.add_argument(
        "routine",
        choices=ROUTINES,
        type=str,
        help="The routine to execute.",
    )

    args = parser.parse_args()

    if args.routine == "detect_humans":
        detect_humans_in_frames()
    else:
        raise ValueError(f"Unknown routine: {args.routine}")
