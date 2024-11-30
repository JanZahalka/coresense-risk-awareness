"""
data.cs_robocup

The CoreSense RoboCup dataset recorded with a social robot.

https://zenodo.org/records/13748065
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CSRoboCup(Dataset):
    """
    The CoreSense RoboCup dataset object.

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
