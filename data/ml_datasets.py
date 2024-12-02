"""
data.ml_datasets

"Umbrella" service functionality for the ML datasets.
"""

import torch.nn as nn
from torch.utils.data import Dataset

from .cs_robocup import CSRoboCup
from .paths import CS_ROBOCUP_ML_RAW_DIR

DATASETS = {"cs_robocup": {"class": CSRoboCup, "dir": CS_ROBOCUP_ML_RAW_DIR}}


def get_dataset(name: str, transform: nn.Module) -> Dataset:
    """
    Get the dataset object by name.
    """

    try:
        dataset_obj = DATASETS[name]["class"](
            root_dir=DATASETS[name]["dir"], transform=transform
        )
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc

    return dataset_obj
