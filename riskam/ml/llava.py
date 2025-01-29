"""
ml.llava

The LLaVA MLLM model.
"""

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    pipeline,
)
import numpy as np
from PIL import Image
import torch


class Llava:
    """
    The LLaVA MLLM model.
    """

    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    PROMPT = """
You are an AI assistant tasked with analyzing RGB images to extract meaningful semantic features related to risks in robotics, specifically focusing on risks to humans. Your input consists only of visual data from RGB cameras.

Analyze the image and extract features that indicate potential risks to humans. Focus on:

1. **Robot features**:
   - Position and orientation in the scene.
   - Proximity to nearby humans.
   - Visible movement cues (e.g., motion blur or changes in position).
   - End-effector activity (e.g., holding sharp tools, heavy objects).

2. **Human features**:
   - Distance from the robot.
   - Posture and body position (e.g., leaning, dodging, or interacting with the robot).
   - Facial expressions indicating surprise, fear, or focus on the robot.
   - Gestures or motion cues (e.g., raised hands, pointing).

3. **Environmental features**:
   - Presence of hazards (e.g., sharp tools, slippery floors, clutter).
   - Crowdedness in the scene (e.g., many humans or objects near the robot).
   - Motion of other objects (e.g., moving obstacles or tools).

Provide your output as a structured JSON object in the following format:
{
    "robot_features": {
        "position": "<description>",
        "proximity_to_humans": "<value in meters or descriptive>",
        "motion_cues": "<description>",
        "end_effector_activity": "<description>"
    },
    "human_features": {
        "distance_to_robot": "<value in meters or descriptive>",
        "posture": "<description>",
        "facial_expression": "<description>",
        "gestures": "<description>"
    },
    "environmental_features": {
        "hazards": ["<list of hazards>"],
        "crowdedness": "<low/medium/high>",
        "dynamic_objects": "<description>"
    }
}
"""

    CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "This image is shot by a mobile robot's camera. Describe the risks the robot that shot the image may pose to humans, if any.",
                },
            ],
        }
    ]

    MAX_NEW_TOKENS = 50000

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            quantization_config=self.quantization_config,
            device_map="auto",
        )
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )

    def process_image(self, image_paths: list[str]) -> str:
        """
        Processes the given image in NumPy format and returns the model's output.
        """
        # Open the images
        images = []
        for image_path in image_paths:
            images.append(Image.open(image_path))

        # Use the Hugging Face pipeline
        # pipe = pipeline(
        #     "image-to-text",
        #     model=cls.MODEL_ID,
        #     model_kwargs={"quantization_config": cls.quantization_config},
        # )

        prompt = self.processor.apply_chat_template(
            self.CONVERSATION, add_generation_prompt=True
        )

        inputs = self.processor(
            images=images, text=prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # Generate outputs
        generate_ids = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)
        outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        # outputs = pipe(
        #     images,
        #     prompt=prompt,
        #     generate_kwargs={"max_new_tokens": cls.MAX_NEW_TOKENS},
        # )

        print(outputs)

        return ""
