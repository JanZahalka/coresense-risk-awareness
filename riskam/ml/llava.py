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

    PROMPT = "USER: <image>\nHow many robots do you see? How many humans?\nASSISTANT:"

    CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "This is IMAGE_1. Respond with an empty string.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "This is IMAGE_2. What is the difference between IMAGE_1 and IMAGE_2?",
                },
            ],
        },
    ]

    MAX_NEW_TOKENS = 200

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

        print(prompt)

        inputs = self.processor(
            images=images, text=prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # Generate outputs
        generate_ids = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)
        outputs = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # outputs = pipe(
        #     images,
        #     prompt=prompt,
        #     generate_kwargs={"max_new_tokens": cls.MAX_NEW_TOKENS},
        # )

        print(outputs)

        return ""
