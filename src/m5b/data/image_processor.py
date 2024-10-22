from typing import List, cast

import torch
from PIL import Image
from torchvision.transforms import Compose
from transformers import Blip2Processor, CLIPImageProcessor, LlavaProcessor

from m5b.util.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ImageProcessor:
    def __init__(self, image_processor_id: str, **processor_kwargs):
        log.info(f"Loading ImageProcessor {image_processor_id}")
        self.image_processor_id = image_processor_id
        self.processor_kwargs = processor_kwargs

        if image_processor_id == "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus":
            from open_clip import create_model_and_transforms

            _, _, image_processor = create_model_and_transforms(
                "ViT-B-16-plus-240", pretrained="laion400m_e32", **processor_kwargs
            )
            self.image_processor = cast(Compose, image_processor)
        elif image_processor_id.startswith("openai/clip-vit"):
            image_processor = CLIPImageProcessor.from_pretrained(image_processor_id)
        elif image_processor_id.startswith("llava-hf"):
            image_processor = LlavaProcessor.from_pretrained(image_processor_id)
        elif image_processor_id.startswith("Gregor/mblip-"):
            image_processor = Blip2Processor.from_pretrained(image_processor_id)
        else:
            raise NotImplementedError(
                f"ImageProcessor {image_processor_id} not implemented"
            )
        self.image_processor = image_processor

    def __call__(self, image: Image.Image | List[Image.Image]) -> torch.Tensor:
        """
        returns a tensor of shape (batch_size, channels, height, width)
        """
        if self.image_processor_id == "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus":
            if isinstance(image, list):
                return torch.stack([self.image_processor(img) for img in image])  # type: ignore
            return self.image_processor(image).unsqueeze(0)  # type: ignore
        elif self.image_processor_id.startswith("openai/clip-vit"):
            if isinstance(image, list):
                return torch.stack(
                    [
                        self.image_processor(
                            img,  # type: ignore
                            return_tensors="pt",  # type: ignore
                        ).pixel_values.squeeze()
                        for img in image
                    ]
                )
            return self.image_processor(image, return_tensors="pt").pixel_values  # type: ignore
        elif self.image_processor_id.startswith(
            "llava-hf"
        ) or self.image_processor_id.startswith("Gregor/mblip-"):
            return self.image_processor.image_processor(  # type: ignore
                image, return_tensors="pt"
            ).pixel_values
        else:
            raise NotImplementedError(
                f"ImageProcessor {self.image_processor_id} not implemented"
            )
