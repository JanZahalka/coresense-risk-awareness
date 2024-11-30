"""
data.paths

The file and directory paths to various datasets.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


# Dataset names
CS_ROBOCUP = "cs_robocup"

# The dir of the ROS datasets (that need to be converted before using them in ML)
ROS_DATA_DIR = ROOT_DIR / "ros_datasets"

# The dir of datasets used for ML
ML_DATA_DIR = ROOT_DIR / "ml_datasets"

# The ROS datasets
CS_ROBOCUP_ROS_DIR = ROS_DATA_DIR / "cs_robocup"


# The ML datasets
CS_ROBOCUP_ML_DIR = ML_DATA_DIR / "cs_robocup"
