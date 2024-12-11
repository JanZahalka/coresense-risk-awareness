"""
extract_ros2_dataset.py

Extracts RGB and depth images from ROS2 datasets.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.data.extract_cs_robocup import extract_cs_robocup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from ROS2 datasets.")

    parser.add_argument(
        "dataset",
        type=str,
        choices=DATASETS.keys(),
        help="The dataset to extract.",
        default="cs_robocup",
    )

    args = parser.parse_args()

    if args.dataset == "cs_robocup":
        extract_cs_robocup()
