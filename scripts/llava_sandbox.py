"""
llava_sandbox.py

A sandbox script for testing the LLaVA model.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.ml.llava import Llava

if __name__ == "__main__":

    IMG_PATHS = [
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223824.506.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223827.468.png",
    ]

    llava = Llava()
    llava.process_image(IMG_PATHS)

    for img_path in IMG_PATHS:
        image = Image.open(img_path)
        image.show()
