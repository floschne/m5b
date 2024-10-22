from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlparse

import open_clip
import torch
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def load_pretrained_mclip_model(
    model_name: str = "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
    cuda_tokenizer: bool = True,
):
    """
    Load a M-CLIP model and return the text and vision encoders, tokenizer, and vision preprocessing function.
    @return: text_encoder, vision_encoder, tokenizer, vison_preprocess
    """
    if not model_name.startswith("M-CLIP"):
        raise NotImplementedError(
            f"Unsupported clip model: {model_name}. Only M-CLIP models are supported"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if cuda_tokenizer and device == "cuda":

        class CudaTokenizer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, *args, **kwargs):
                return self.tokenizer(*args, **kwargs, truncation=True).to(device)

        tokenizer = CudaTokenizer(tokenizer)

    if "XLM-Roberta-Large-Vit-B-16Plus" in model_name:
        vision_encoder, _, vison_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-plus-240", pretrained="laion400m_e32"
        )
    else:
        raise NotImplementedError()

    text_encoder.to(device)
    text_encoder.eval()

    vision_encoder.to(device)
    vision_encoder.eval()

    return text_encoder, vision_encoder, tokenizer, vison_preprocess


class torch_default_dtype:
    """
    Context manager for setting the default torch dtype.
    """

    def __init__(self, dtype: torch.dtype | str | None) -> None:
        self.dtype = dtype

    def __enter__(self) -> Any:
        self.dtype_orig = torch.get_default_dtype()
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Optional[bool]:
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype_orig)


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
