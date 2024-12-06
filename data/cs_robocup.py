"""
data.cs_robocup

The CoreSense RoboCup dataset recorded with a social robot.

https://zenodo.org/records/13748065
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import YolosImageProcessor, YolosForObjectDetection
from tqdm import tqdm

from paths import CS_ROBOCUP_ML_DIR, CS_ROBOCUP_ML_RAW_DIR


# CoreSense Robocup ML dataset JSONs
CS_ROBOCUP_N_HUMANS_PATH = CS_ROBOCUP_ML_DIR / "n_humans.json"
CS_ROBOCUP_BBOXES_PATH = CS_ROBOCUP_ML_DIR / "bboxes.json"
CS_ROBOCUP_DISTANCES_PATH = CS_ROBOCUP_ML_DIR / "distances.json"


# Script constants
ROUTINES = ["detect_humans", "visualize", "compute_distances"]

# Invalid depth value
INVALID_DEPTH = -1


class CSRoboCup(Dataset):
    """
    The CoreSense RoboCup PyTorch dataset object.

    Currently implemented as a SUPERVISED dataset with run identifiers as labels. That said,
    it's an unsupervised dataset for ML intents and purposes.
    """

    TASK_TYPES = ["humans"]  # 1 = at least one human, 0 = no humans in the frame

    def __init__(self, root_dir: str, task_type: str, transform=None):
        # Initialize the dataset's variables
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.classes = None
        self.class_to_idx = None

        if task_type == "humans":
            self._load_data_humans()
        if task_type not in self.TASK_TYPES:
            raise ValueError(f"Unknown task type: {task_type}")

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

    def _load_data_humans(self) -> None:
        """
        Loads the data for the "humans" task.
        """
        with open(CS_ROBOCUP_N_HUMANS_PATH) as f:
            n_humans = json.load(f)

        # Establish class labels
        self.classes = ["no_humans", "humans"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Fill the dataset
        for rb in range(1, 9):
            rb_dir = self.root_dir / f"RB_0{rb}" / "rgb"

            for img_path in rb_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(1 if n_humans[f"RB_0{rb}"][img_path.name] > 0 else 0)


def detect_humans_in_frames() -> None:
    """
    Go over the CoreSense Robocup dataset and for each frame, extract the number of humans in it
    and the corresponding bounding boxes.
    """

    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    if CS_ROBOCUP_N_HUMANS_PATH.exists():
        n_humans = json.loads(CS_ROBOCUP_N_HUMANS_PATH.read_text())
    else:
        n_humans = {}
        CS_ROBOCUP_N_HUMANS_PATH.write_text(json.dumps(n_humans, indent=4))

    if CS_ROBOCUP_BBOXES_PATH.exists():
        bboxes = json.loads(CS_ROBOCUP_BBOXES_PATH.read_text())
    else:
        bboxes = {}
        CS_ROBOCUP_BBOXES_PATH.write_text(json.dumps(bboxes, indent=4))

    # Iterate over the runs
    for rb in range(1, 9):
        print(f"+++ PROCESSING RUN RB_0{rb} +++")

        rb_label = f"RB_0{rb}"

        # Establish the RB in the dicts
        if f"RB_0{rb}" in n_humans and f"RB_0{rb}" in bboxes:
            print("This run has already been processed. Skipping...")
            continue

        n_humans[rb_label] = {}
        bboxes[rb_label] = {}

        rb_rgb_dir = CS_ROBOCUP_ML_RAW_DIR / f"RB_0{rb}" / "rgb"

        for img_path in tqdm(list(rb_rgb_dir.glob("*.npy"))):
            # Establish the entry in the dicts
            n_humans[rb_label][img_path.name] = 0
            bboxes[rb_label][img_path.name] = []

            # Load the image
            try:
                image_array = np.load(img_path)
                image = Image.fromarray(np.uint8(image_array)).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue

            # Process the image
            inputs = processor(image, return_tensors="pt")
            outputs = model(**inputs)

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

                n_humans[rb_label][img_path.name] += 1
                bboxes[rb_label][img_path.name].append(box.tolist())

                box = [round(i, 2) for i in box.tolist()]
                """
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                """

        # Save the results
        CS_ROBOCUP_N_HUMANS_PATH.write_text(json.dumps(n_humans, indent=4))
        CS_ROBOCUP_BBOXES_PATH.write_text(json.dumps(bboxes, indent=4))


def _find_closest_rgb_image(
    rgb_images_sorted: list[float], depth_image_timestamp: float
) -> str:
    """
    Finds the closest RGB image based on depth image timestamp. Utilizes binary halving.
    """

    left = 0
    right = len(rgb_images_sorted) - 1

    while left <= right:
        mid = (left + right) // 2

        if rgb_images_sorted[mid] == depth_image_timestamp:
            return f"{rgb_images_sorted[mid]}.png"
        elif rgb_images_sorted[mid] < depth_image_timestamp:
            left = mid + 1
        else:
            right = mid - 1

    return f"{rgb_images_sorted[right]}.npy"


def compute_human_distances() -> None:
    """
    Computes the distances between the robot and the detected human. The method is
    median bbox value.
    """

    bboxes = json.loads(CS_ROBOCUP_BBOXES_PATH.read_text())
    distances = {}

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        distances[rb_label] = {}

        rb_depth_dir = CS_ROBOCUP_ML_RAW_DIR / f"RB_0{rb}" / "depth"
        rgb_images = [
            float(".".join(i.split(".")[:-1]))
            for i in sorted(list(bboxes[rb_label].keys()))
        ]

        for img_path in tqdm(sorted(list(rb_depth_dir.glob("*.npy")))):
            try:
                depth_image = np.load(img_path)
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue

            # Find the closest RGB image
            closest_rgb_image = _find_closest_rgb_image(
                rgb_images, float(img_path.stem)
            )

            # Some images may not have bounding boxes (no humans in the frame)
            if closest_rgb_image not in bboxes[rb_label]:
                continue

            distances[rb_label][img_path.name] = []

            for box in bboxes[rb_label][closest_rgb_image]:
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Extract depth values within the bounding box
                sub_image = depth_image[y1:y2, x1:x2]

                # Compute the median depth value, ignoring NaNs
                if np.isnan(sub_image).all():
                    median_depth = np.nan
                else:
                    median_depth = np.nanmedian(sub_image)

                if np.isnan(median_depth):
                    median_depth = INVALID_DEPTH

                distances[rb_label][img_path.name].append(float(median_depth))

    CS_ROBOCUP_DISTANCES_PATH.write_text(json.dumps(distances, indent=4))


def draw_bbox(image_cv2, box):
    """Draw bounding boxes on the image."""
    x1, y1, x2, y2 = box
    cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return image_cv2


def visualize_humans_bboxes() -> None:
    """
    Visualize the bounding boxes of the detected humans in the CoreSense RoboCup dataset.
    """

    bboxes = json.loads(CS_ROBOCUP_BBOXES_PATH.read_text())

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"
        rb_rgb_dir = CS_ROBOCUP_ML_RAW_DIR / f"RB_0{rb}" / "rgb"

        for img_path in tqdm(sorted(list(rb_rgb_dir.glob("*.npy")))):
            try:
                image_array = np.load(img_path)
                image = Image.fromarray(np.uint8(image_array)).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            for box in bboxes[rb_label][img_path.name]:
                image = draw_bbox(image_cv2, box)

            # Display the image
            cv2.imshow("Object Detection Viewer", image_cv2)

            # Handle key presses
            key = cv2.waitKey(0)
            if key == 27:  # Escape key
                break
            elif key == ord("d"):  # Next image
                current_index = (current_index + 1) % len(image_files)
            elif key == ord("a"):  # Previous image
                current_index = (current_index - 1) % len(image_files)


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
    elif args.routine == "visualize":
        visualize_humans_bboxes()
    elif args.routine == "compute_distances":
        compute_human_distances()
    else:
        raise ValueError(f"Unknown routine: {args.routine}")
