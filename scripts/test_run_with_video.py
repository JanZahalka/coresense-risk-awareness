"""
test_run_with_video.py

Runs the risk awareness model on a sequence of images, creating 1) a video from the images,
and 2) a video with the risk score overlay.
"""

import argparse
from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.ml import featextr
from riskam import score, video

TEST_RESULTS_DIR = Path("test_results")
RISK_SCORE_MASTER_DIR = TEST_RESULTS_DIR / "risk_scores"
VIDEO_DIR = TEST_RESULTS_DIR / "videos"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the risk awareness model on a sequence of images."
    )
    parser.add_argument(
        "dataset", type=str, choices=DATASETS.keys(), help="The dataset to process."
    )
    parser.add_argument("run", type=str, help="The run to process.")

    args = parser.parse_args()

    # Establish the dirs
    IMG_DIR = DATASETS[args.dataset]["img_dir"]

    if args.dataset == "cs_robocup_2023":
        IMG_DIR = IMG_DIR / args.run / "rgb"

    VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    risk_score_dir = RISK_SCORE_MASTER_DIR / args.dataset / args.run
    risk_score_dir.mkdir(exist_ok=True, parents=True)

    # Perform the risk awareness analysis
    for img_path in tqdm(sorted(IMG_DIR.iterdir())):
        with Image.open(img_path) as image:
            features, visualization = featextr.extract_human_risk_awareness_features(
                image, img_path, visualize=True
            )
            risk_score = score.risk_awareness_score(features)
            risk_score_image = score.risk_score_image_overlay(visualization, risk_score)

            risk_score_output_path = risk_score_dir / Path(img_path).name

            if risk_score_image is not None:
                risk_score_image.save(risk_score_output_path)

    # Process the videos
    video.images_to_video(
        risk_score_dir,
        VIDEO_DIR / f"{args.dataset}_{args.run}_risk.avi",
        image_extension="png",
        fps=10,
    )
    video.images_to_video(
        IMG_DIR, VIDEO_DIR / f"{args.dataset}_{args.run}_raw.avi", image_extension="png"
    )
