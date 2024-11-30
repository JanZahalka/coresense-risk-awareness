"""
ml.train

The training script.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).parent.parent))

from data.ml_datasets import DATASETS  # pylint: disable=wrong-import-position
from ml.autoencoder import (  # pylint: disable=wrong-import-position
    VariationalAutoEncoder,
)

MODELS = {"vae": VariationalAutoEncoder}

ML_MODELS_DIR = Path(__file__).parent.parent / "ml_models"


def train(
    model_type: str,
    dataset: str,
    batch_size: int,
    n_epochs: int,
    n_dataloader_workers: int,
) -> None:
    """
    Train a model.
    """

    # Train the model
    model = MODELS[model_type].train_model(
        dataset, batch_size, n_epochs, n_dataloader_workers
    )

    # Store it
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_fname = ML_MODELS_DIR / f"{dataset}-{model_type}-{n_epochs}ep.pt"
    torch.save(model, model_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "model_type",
        choices=MODELS.keys(),
        type=str,
        help="The type of the model to train.",
    )
    parser.add_argument(
        "dataset", choices=DATASETS.keys(), type=str, help="The dataset to train on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="The training batch size."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="The number of epochs."
    )
    parser.add_argument(
        "--n_dataloader_workers",
        type=int,
        default=1,
        help="The number of CPU workers for the DataLoader.",
    )

    args = parser.parse_args()

    train(
        args.model_type,
        args.dataset,
        args.batch_size,
        args.n_epochs,
        args.n_dataloader_workers,
    )
