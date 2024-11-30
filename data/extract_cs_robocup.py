"""
extract_cs_robocup.py

Extract data from the CoreSense Robocup social image dataset in a ML-friendly format.
"""

from pathlib import Path

import cv2
from sensor_msgs.msg import Image  # pylint: disable=import-error
from cv_bridge import CvBridge  # pylint: disable=import-error
import numpy as np
from rclpy.serialization import deserialize_message  # pylint: disable=import-error
from rosbag2_py import (  # pylint: disable=import-error
    SequentialReader,
    StorageOptions,
    ConverterOptions,
)

from paths import CS_ROBOCUP_ROS_DIR, CS_ROBOCUP_ML_DIR

TOPIC_RGB_IMAGE = "/xtion/rgb/image_raw"
TOPIC_DEPTH_IMAGE = "/xtion/depth/image_raw"

EXTRACTION_PARAMS = {
    TOPIC_RGB_IMAGE: {"encoding": "rgb8", "dir": "rgb"},
    TOPIC_DEPTH_IMAGE: {"encoding": "passthrough", "dir": "depth"},
}


def extract_cs_robocup() -> None:
    """
    The main extractor function.
    """

    for rb in range(1, 9):
        bag_dir = CS_ROBOCUP_ROS_DIR / f"RB_0{rb}" / f"RB_0{rb}"

        output_dir = CS_ROBOCUP_ML_DIR / f"RB_0{rb}"
        rgb_output_dir = output_dir / "rgb"
        depth_output_dir = output_dir / "depth"

        if not bag_dir.exists() or (
            rgb_output_dir.exists() and depth_output_dir.exists()
        ):
            continue

        rgb_output_dir.mkdir(parents=True, exist_ok=True)
        depth_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting data from RB_0{rb}...")

        for bag_path in bag_dir.glob("*.db3"):
            storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )

            reader = SequentialReader()
            reader.open(storage_options, converter_options)

            # topics = reader.get_all_topics_and_types()
            bridge = CvBridge()

            # for topic in topics:
            #    print(f"Found topic: {topic.name} of type {topic.type}")

            while reader.has_next():
                (topic_name, data, _) = reader.read_next()

                if topic_name in [TOPIC_RGB_IMAGE, TOPIC_DEPTH_IMAGE]:
                    # Deserialize
                    img_msg = deserialize_message(data, Image)

                    try:
                        cv_image = bridge.imgmsg_to_cv2(
                            img_msg,
                            desired_encoding=EXTRACTION_PARAMS[topic_name]["encoding"],
                        )

                        if topic_name == TOPIC_DEPTH_IMAGE:
                            cv_image = cv2.normalize(
                                cv_image, None, 0, 255, cv2.NORM_MINMAX
                            )
                            cv_image = np.uint8(cv_image)

                        timestamp = (
                            img_msg.header.stamp.sec
                            + img_msg.header.stamp.nanosec * 1e-9
                        )

                        image_path = (
                            output_dir
                            / f"{EXTRACTION_PARAMS[topic_name]['dir']}/{timestamp:.3f}.png"
                        )
                        cv2.imwrite(str(image_path), cv_image)
                        # print(f"Saved image: {image_path}")

                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Error processing image: {e}")


if __name__ == "__main__":
    extract_cs_robocup()
