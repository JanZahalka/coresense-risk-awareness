"""
ml.llava

The LLaVA MLLM model.
"""

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    pipeline,
)
import numpy as np
from PIL import Image
import torch


class Llava:
    """
    The LLaVA MLLM model.
    """

    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    PROMPT = "USER: <image>\nWhat letters do you see?\nASSISTANT:"

    MAX_NEW_TOKENS = 200

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            quantization_config=self.quantization_config,
            device_map="auto",
        )

    @classmethod
    def process_image(cls, np_image_path: str) -> str:
        """
        Processes the given image in NumPy format and returns the model's output.
        """
        np_image = np.load(np_image_path)
        np_image = np_image.astype(np.uint8)  # Cast as image format

        # Convert to PIL
        image = Image.fromarray(np_image)

        # Use the Hugging Face pipeline
        pipe = pipeline(
            "image-to-text",
            model=cls.MODEL_ID,
            model_kwargs={"quantization_config": cls.quantization_config},
        )

        # Generate outputs
        outputs = pipe(
            image,
            prompt=cls.PROMPT,
            generate_kwargs={"max_new_tokens": cls.MAX_NEW_TOKENS},
        )

        print(outputs)
        print()
        print(outputs[0]["generated_text"])

        return ""
