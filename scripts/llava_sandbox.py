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
    NP_IMAGE_PATH = "ml_datasets/cs_robocup/raw_dataset/RB_01/rgb/1537223824.506.npy"
    Llava.process_image(NP_IMAGE_PATH)

    np_image = np.load(NP_IMAGE_PATH)
    image = Image.fromarray(np_image)

    image.show()
