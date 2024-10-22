import base64
import io
import os
import re
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import fire
import pandas as pd
import srsly
import torch
from lightning import seed_everything
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPImageProcessor,
    LlamaTokenizer,
    LlavaForConditionalGeneration,
)
from transformers.generation import GenerationConfig

from m5b.data.image_processor import ImageProcessor
from m5b.data.lit_m5b_vgr import LitM5BVGRWDS
from m5b.data.lit_m5b_vlod import LitM5BVLODWDS
from m5b.data.lit_marvl_wds import LitMaRVLWDS
from m5b.data.lit_maxm_wds import LitMaXMWDS
from m5b.data.lit_xflickrco_wds import LitxFlickrCOMWDS
from m5b.data.lit_xgqa_wds import LitXGQAWDS
from m5b.data.lit_xm3600_wds import LitXM3600WDS
from m5b.data.lit_xvnli_wds import LitXVNLIWDS
from m5b.data.text_processor import TextProcessor
from m5b.util.caption_prompter import CaptionPrompter
from m5b.util.evaluation import (
    generated_caption_evaluation,
    generated_label_classification_evaluation,
)

if TYPE_CHECKING:
    from m5b.model.api_models import (
        GeminiProVisionGoogleStudioModel,
        GeminiProVisionVertexAIModel,
        GPT4VisionOpenAIModel,
    )


class Batch(TypedDict):
    prompts: List[str]
    gold_text: List[str] | List[List[str]]
    images: List[Image.Image]
    languages: List[str]
    sample_ids: List[str]
    keys: List[str]

    @staticmethod  # type: ignore
    def fields():
        return Batch.__dict__["__annotations__"].keys()


xgqa_languages = {
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "id": "Indonesian",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}

xm3600_languages = {
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mi": "Maori",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "quz": "Cusco Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

marvl_languages = {
    "tr": "Turkish",
    "id": "Indonesian",
    "ta": "Tamil",
    "zh": "Chinese",
    "sw": "Swahili",
}
marvl_english_translations = {
    "en-tr": "English_From_Turkish",
    "en-id": "English_From_Indonesian",
    "en-ta": "English_From_Tamil",
    "en-zh": "English_From_Chinese",
    "en-sw": "English_From_Swahili",
}

xvnli_languages = {
    "ar": "Arabic",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ru": "Russian",
}

maxm_languages = {
    "en": "English",
    "fr": "French",
    "hi": "Hindi",
    "iw": "Hebrew",
    "ro": "Romanian",
    "th": "Thai",
    "zh": "Chinese",
}

xflickrco_languages = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "id": "Indonesian",
    "ja": "Japanese",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
}

m5b_vlod_languages = m5b_vgr_languages = {
    "am": "Amharic",
    "ber": "Berber",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "fil": "Filipino",
    "ha": "Hausa",
    "hi": "Hindi",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "zu": "Zulu",
}

all_langs = {
    **xgqa_languages,
    **xm3600_languages,
    **marvl_languages,
    **marvl_english_translations,
    **xvnli_languages,
    **maxm_languages,
    **xflickrco_languages,
    **m5b_vlod_languages,
}


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def _check_or_create_prompt_template(
    dataset: str, model_id: str, prompt_template: str | None
) -> str:
    match dataset:
        case "marvl":
            if prompt_template is None:
                prompt_template = "Based on the two images, is it correct to say ”{HYPOTHESIS}”? Yes or no? One word answer in English:"
            if "{HYPOTHESIS}" not in prompt_template:
                raise ValueError(
                    "Prompt template for MaRVL dataset must contain {HYPOTHESIS}!"
                )
        case "xvnli":
            if prompt_template is None:
                prompt_template = "Is it guaranteed true that ”{HYPOTHESIS}”? Yes, no, or maybe? One word answer in English:"
            if "{HYPOTHESIS}" not in prompt_template:
                raise ValueError(
                    "Prompt template for XVNLI dataset must contain {HYPOTHESIS}!"
                )
        case "xgqa":
            if prompt_template is None:
                prompt_template = "Question: {QUESTION} Short answer in English:"
            if "{QUESTION}" not in prompt_template:
                raise ValueError(
                    "Prompt template for XGQA dataset must contain {QUESTION}!"
                )
        case "maxm":
            if prompt_template is None:
                prompt_template = "Question: {QUESTION} Short answer in {LANGUAGE}:"
            if (
                "{QUESTION}" not in prompt_template
                or "{LANGUAGE}" not in prompt_template
            ):
                raise ValueError(
                    "Prompt template for MaXM dataset must contain {QUESTION} and {LANGUAGE}!"
                )
        case "xflickrco":
            if prompt_template is None:
                prompt_template = "Brief caption in {LANGUAGE}:"
            if "{LANGUAGE}" not in prompt_template:
                raise ValueError(
                    "Prompt template for xFlickrCo dataset must {LANGUAGE}!"
                )
        case "xm3600":
            if prompt_template is None:
                prompt_template = "{PROMPT}"  # this gets passed to the caption prompter
            pass
        case "m5b_vgr":
            if prompt_template is None:
                prompt_template = "Based on the two images, is it correct to say ”{HYPOTHESIS}”? Yes or no? One word answer in English:"
            if "{HYPOTHESIS}" not in prompt_template:
                raise ValueError(
                    "Prompt template for M5B VGR dataset must contain {HYPOTHESIS}!"
                )
        case "m5b_vlod":
            if prompt_template is None:
                prompt_template = (
                    "Based on the {NUM_IMGS} images ordered from top-left to bottom-right, "
                    "which image does not match the hypothesis ”{HYPOTHESIS}”? Choose one from {CHOICES} and only output a single letter:"
                )
            if "{NUM_IMGS}" not in prompt_template:
                raise ValueError(
                    "Prompt template for M5B VLOD must contain '{NUM_IMGS}'"
                )
            if "{HYPOTHESIS}" not in prompt_template:
                raise ValueError(
                    "Prompt template for M5B VLOD must contain '{HYPOTHESIS}'"
                )
            if "{CHOICES}" not in prompt_template:
                raise ValueError(
                    "Prompt template for M5B VLOD must contain '{CHOICES}'"
                )
        case _:
            raise ValueError(
                f"There is no prompt template for dataset {dataset} defined!"
            )

    if model_id.startswith("llava-hf/"):
        prompt_template = f"USER: <image>\n{prompt_template}\nASSISTANT:"
    elif model_id.startswith("THUDM/cogvlm-chat-hf"):
        prompt_template = prompt_template
    elif model_id.startswith("Gregor/mblip-"):
        prompt_template = prompt_template
    elif model_id.startswith("liuhaotian/llava-"):
        # we use the liuhaotian llava code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("openbmb/OmniLMM-"):
        # we use the OmniLMM code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("openbmb/MiniCPM-V"):
        # we use the MiniCPM code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("Qwen/Qwen-VL-Chat"):
        # we use the Qwen-VL-Chat code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("01-ai/Yi-VL-"):
        # we use the Yi VL code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("OpenGVLab/InternVL-Chat-"):
        # we use the Intern VL code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("gpt-4"):
        # we use the GPT4V code to handle correct prompting
        prompt_template = prompt_template
    elif model_id.startswith("gemini-pro-vision"):
        # we use the Gemini code to handle correct prompting
        prompt_template = prompt_template
    else:
        raise ValueError(f"There is no prompt template for model {model_id} defined!")

    return prompt_template


def _load_marvl(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
    use_stacked_images: Literal["vertically", "horizontally"] = "horizontally",
    use_english_translation: Literal["en-tr", "en-id", "en-ta", "en-zh", "en-sw", "all"]
    | None = None,
    only_use_translation: bool = False,
) -> LitMaRVLWDS:
    languages = {k: v for k, v in marvl_languages.items() if k in langs}
    if not only_use_translation:
        print(f"Loading MaRVL dataset: {data_base_path} with languages: {languages}")
    print(
        f"Loading MaRVL dataset: {data_base_path} with English Translation: {use_english_translation}"
    )

    marvl = LitMaRVLWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        use_english_translation=use_english_translation,
        only_use_translation=only_use_translation,
        use_stacked_images=use_stacked_images,
        no_collate=True,
    )
    marvl.prepare_data()

    return marvl


def _load_m5b_vgr(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
    use_stacked_images: Literal["vertically", "horizontally"] = "horizontally",
) -> LitM5BVGRWDS:
    languages = {k: v for k, v in m5b_vgr_languages.items() if k in langs}

    m5b_vgr = LitM5BVGRWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        use_stacked_images=use_stacked_images,
        no_collate=True,
    )
    m5b_vgr.prepare_data()

    return m5b_vgr


def _load_m5b_vlod(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
    use_stacked_images: Literal["vertically", "horizontally", "two_cols"] = "two_cols",
) -> LitM5BVLODWDS:
    languages = {k: v for k, v in m5b_vlod_languages.items() if k in langs}

    m5b_vlod = LitM5BVLODWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        use_stacked_images=use_stacked_images,
        no_collate=True,
    )
    m5b_vlod.prepare_data()

    return m5b_vlod


def _load_xvnli(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
) -> LitXVNLIWDS:
    languages = {k: v for k, v in xvnli_languages.items() if k in langs}
    print(f"Loading XVNLI dataset: {data_base_path} with languages: {languages}")

    xvnli = LitXVNLIWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        no_collate=True,
    )
    xvnli.prepare_data()

    return xvnli


def _load_xgqa(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
) -> LitXGQAWDS:
    languages = {k: v for k, v in xgqa_languages.items() if k in langs}
    print(f"Loading XGQA dataset: {data_base_path} with languages: {languages}")

    xgqa = LitXGQAWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        no_collate=True,
    )
    xgqa.prepare_data()

    return xgqa


@lru_cache(maxsize=1)
def __get_caption_prompter(prompt_template: str) -> CaptionPrompter:
    return CaptionPrompter(
        prompt_template=prompt_template if len(prompt_template) > 0 else None,
        colon=True,
    )


def _load_xm3600(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    image_processor: ImageProcessor | None = None,
    batch_size: int = 5,
    num_workers: int = 1,
) -> LitXM3600WDS:
    languages = {k: v for k, v in xm3600_languages.items() if k in langs}
    print(f"Loading XM3600 dataset: {data_base_path} with languages: {languages}")

    xm3600 = LitXM3600WDS(
        data_base_path=data_base_path,
        languages=languages,
        image_processor=image_processor,
        caption_prompter=__get_caption_prompter(
            prompt_template
        ),  # TODO: make caption prompter optional
        batch_size=batch_size,
        num_workers=num_workers,
        no_collate=True,
    )
    xm3600.prepare_data()

    return xm3600


def _load_maxm(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
) -> LitMaXMWDS:
    languages = {k: v for k, v in maxm_languages.items() if k in langs}
    print(f"Loading MaXM dataset: {data_base_path} with languages: {languages}")

    maxm = LitMaXMWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        no_collate=True,
    )
    maxm.prepare_data()
    return maxm


def _load_xflickrco(
    data_base_path: Path,
    langs: list,
    prompt_template: str,
    batch_size: int = 5,
    num_workers: int = 1,
) -> LitxFlickrCOMWDS:
    languages = {k: v for k, v in xflickrco_languages.items() if k in langs}
    print(f"Loading xFlickrCo dataset: {data_base_path} with languages: {languages}")

    xflickrco = LitxFlickrCOMWDS(
        data_base_path=data_base_path,
        languages=languages,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        no_collate=True,
    )
    xflickrco.prepare_data()

    return xflickrco


def load_dataset(
    dataset: Literal[
        "marvl",
        "xgqa",
        "xm3600",
        "xvnli",
        "maxm",
        "xflickrco",
        "m5b_vgr",
        "m5b_vlod",
    ],
    data_base_path: Path,
    prompt_template: str,
    langs: list | None = None,
    batch_size: int = 5,
    num_workers: int = 1,
    use_stacked_images: Literal[
        "vertically", "horizontally", "two_cols"
    ] = "horizontally",
    marvl_use_english_translation: Literal[
        "en-tr", "en-id", "en-ta", "en-zh", "en-sw", "all"
    ]
    | None = None,
    marvl_only_use_translation: bool = False,
) -> (
    LitMaRVLWDS
    | LitXGQAWDS
    | LitXM3600WDS
    | LitXVNLIWDS
    | LitMaXMWDS
    | LitxFlickrCOMWDS
    | LitM5BVGRWDS
    | LitM5BVLODWDS
):
    if dataset == "marvl":
        if langs is None:
            langs = list(marvl_languages.keys())
        return _load_marvl(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
            use_stacked_images=use_stacked_images,
            use_english_translation=marvl_use_english_translation,
            only_use_translation=marvl_only_use_translation,
        )
    elif dataset == "xgqa":
        if langs is None:
            langs = list(xgqa_languages.keys())
        return _load_xgqa(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "xm3600":
        if langs is None:
            langs = list(xm3600_languages.keys())
        return _load_xm3600(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "xvnli":
        if langs is None:
            langs = list(xvnli_languages.keys())
        return _load_xvnli(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "maxm":
        if langs is None:
            langs = list(maxm_languages.keys())
        return _load_maxm(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "xflickrco":
        if langs is None:
            langs = list(xflickrco_languages.keys())
        return _load_xflickrco(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == "m5b_vgr":
        if langs is None:
            langs = list(m5b_vgr_languages.keys())
        return _load_m5b_vgr(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
            use_stacked_images=use_stacked_images,
        )
    elif dataset == "m5b_vlod":
        if langs is None:
            langs = list(m5b_vlod_languages.keys())
        return _load_m5b_vlod(
            data_base_path=data_base_path,
            langs=langs,
            prompt_template=prompt_template,
            batch_size=batch_size,
            num_workers=num_workers,
            use_stacked_images=use_stacked_images,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported!")


def _decode_bytes_to_str(x: str | bytes) -> str | Any:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    else:
        return x  # type: ignore


def _build_batch(
    *,
    prompts: List[str],
    gold_text: List[str] | List[List[str]],
    images: List[Image.Image],
    languages: List[str],
    sample_ids: List[str],
    keys: List[str],
) -> Batch:
    return {
        "prompts": prompts,
        "gold_text": gold_text,
        "images": images,
        "languages": languages,
        "sample_ids": sample_ids,
        "keys": keys,
    }


def _xm3600_build_batch(
    *,
    batch: list,
    prompt_template: str,
) -> Batch:
    images = batch[0]
    captions = list(map(_decode_bytes_to_str, batch[1]))
    langs = list(map(_decode_bytes_to_str, batch[2]))
    image_ids = list(map(_decode_bytes_to_str, batch[3]))
    keys = list(map(_decode_bytes_to_str, batch[4]))

    # skip duplicate images based on image_id...
    # Since the images are exactly the same, we can just skip the duplicates
    # because they will lead to the same captions or outputs
    unique_images = []
    unique_captions = []
    unique_langs = []
    unique_image_ids = []
    unique_keys = []
    seen_image_ids = set()
    for i, image_id in enumerate(image_ids):
        if image_id not in seen_image_ids:
            unique_images.append(images[i])
            unique_captions.append(captions[i])
            unique_langs.append(langs[i])
            unique_image_ids.append(image_ids[i])
            unique_keys.append(keys[i])
            seen_image_ids.add(image_id)

    unique_languages = [xm3600_languages[lang] for lang in unique_langs]
    prompts = [
        __get_caption_prompter(prompt_template).generate_prompt(
            lang=xm3600_languages[lang]
        )
        for lang in unique_langs
    ]

    return _build_batch(
        prompts=prompts,
        gold_text=unique_captions,
        images=unique_images,
        languages=unique_languages,
        sample_ids=unique_image_ids,
        keys=unique_keys,
    )


def _marvl_build_batch(
    *,
    batch: list,
    prompt_template: str,
    use_stacked_images: Literal["vertically", "horizontally"],
    translation: bool,
) -> Batch:
    if use_stacked_images == "vertically":
        images = batch[2]
    elif use_stacked_images == "horizontally":
        images = batch[3]
    else:
        raise ValueError(f"Invalid value for stacked_images: {use_stacked_images}")

    hypotheses = list(map(_decode_bytes_to_str, batch[4]))
    label_strs = list(map(_decode_bytes_to_str, batch[5]))
    langs = list(map(_decode_bytes_to_str, batch[7]))
    sample_ids = list(map(_decode_bytes_to_str, batch[9]))
    en_translations = list(map(_decode_bytes_to_str, batch[10]))
    keys = list(map(_decode_bytes_to_str, batch[13]))

    if translation:
        languages = [marvl_english_translations[f"en-{lang}"] for lang in langs]
        prompts = [prompt_template.format(HYPOTHESIS=h) for h in en_translations]
    else:
        languages = [marvl_languages[lang] for lang in langs]
        prompts = [prompt_template.format(HYPOTHESIS=h) for h in hypotheses]

    return _build_batch(
        prompts=prompts,
        gold_text=label_strs,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def _m5b_vgr_build_batch(
    *,
    batch: list,
    prompt_template: str,
    use_stacked_images: Literal["vertically", "horizontally"],
) -> Batch:
    if use_stacked_images == "vertically":
        images = batch[0]
    elif use_stacked_images == "horizontally":
        images = batch[1]
    else:
        raise ValueError(f"Invalid value for stacked_images: {use_stacked_images}")

    hypotheses = list(map(_decode_bytes_to_str, batch[2]))
    label_strs = list(map(_decode_bytes_to_str, batch[3]))
    topics = list(map(_decode_bytes_to_str, batch[4]))
    languages = list(map(_decode_bytes_to_str, batch[5]))
    sample_ids = list(map(_decode_bytes_to_str, batch[6]))
    en_translations = list(map(_decode_bytes_to_str, batch[7]))
    keys = list(map(_decode_bytes_to_str, batch[8]))

    languages = [m5b_vgr_languages[lang] for lang in languages]
    prompts = [prompt_template.format(HYPOTHESIS=h) for h in hypotheses]

    return _build_batch(
        prompts=prompts,
        gold_text=label_strs,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def _m5b_vlod_build_batch(
    *,
    batch: list,
    prompt_template: str,
    use_stacked_images: Literal["vertically", "horizontally", "two_cols"],
    choices_type: Literal["alpha", "numeric"],
) -> Batch:
    if use_stacked_images == "vertically":
        images = batch[0]
    elif use_stacked_images == "horizontally":
        images = batch[1]
    elif use_stacked_images == "two_cols":
        images = batch[2]
    else:
        raise ValueError(f"Invalid value for stacked_images: {use_stacked_images}")

    hypotheses = list(map(_decode_bytes_to_str, batch[3]))
    label_strs = list(map(_decode_bytes_to_str, batch[4]))
    topics = list(map(_decode_bytes_to_str, batch[5]))
    languages = list(map(_decode_bytes_to_str, batch[6]))
    sample_ids = list(map(_decode_bytes_to_str, batch[7]))
    keys = list(map(_decode_bytes_to_str, batch[8]))

    if choices_type == "alpha":
        m = dict(zip(map(lambda i: str(i), range(1, 6)), "ABCDE"))
        label_strs = [m[label] for label in label_strs]
    languages = [m5b_vlod_languages[lang] for lang in languages]
    prompts = [prompt_template.format(HYPOTHESIS=h) for h in hypotheses]

    return _build_batch(
        prompts=prompts,
        gold_text=label_strs,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def _xgqa_build_batch(
    *,
    batch: list,
    prompt_template: str,
) -> Batch:
    images = batch[0]
    questions = list(map(_decode_bytes_to_str, batch[1]))
    answers = list(map(_decode_bytes_to_str, batch[2]))
    image_ids = list(map(_decode_bytes_to_str, batch[3]))
    langs = list(map(_decode_bytes_to_str, batch[4]))
    keys = list(map(_decode_bytes_to_str, batch[5]))

    languages = [xgqa_languages[lang] for lang in langs]
    prompts = [prompt_template.format(QUESTION=q) for q in questions]

    return _build_batch(
        prompts=prompts,
        gold_text=answers,
        images=images,
        languages=languages,
        sample_ids=image_ids,
        keys=keys,
    )


def _xvnli_build_batch(
    *,
    batch: list,
    prompt_template: str,
) -> Batch:
    images = batch[0]
    hypotheses = list(map(_decode_bytes_to_str, batch[1]))
    label_strs = list(map(_decode_bytes_to_str, batch[2]))
    langs = list(map(_decode_bytes_to_str, batch[4]))
    sample_ids = list(map(_decode_bytes_to_str, batch[5]))
    keys = list(map(_decode_bytes_to_str, batch[6]))

    languages = [xvnli_languages[lang] for lang in langs]
    prompts = [prompt_template.format(HYPOTHESIS=h) for h in hypotheses]

    return _build_batch(
        prompts=prompts,
        gold_text=label_strs,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def _maxm_build_batch(
    *,
    batch: list,
    prompt_template: str,
) -> Batch:
    images = batch[0]
    questions = list(map(_decode_bytes_to_str, batch[1]))
    answers = list(map(lambda ans: _decode_bytes_to_str(ans).split(","), batch[2]))
    langs = list(map(_decode_bytes_to_str, batch[3]))
    sample_ids = list(map(_decode_bytes_to_str, batch[4]))
    keys = list(map(_decode_bytes_to_str, batch[5]))

    languages = [maxm_languages[lang] for lang in langs]
    prompts = [
        prompt_template.format(QUESTION=question, LANGUAGE=language)
        for question, language in zip(questions, languages)
    ]

    return _build_batch(
        prompts=prompts,
        gold_text=answers,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def _xflickrco_build_batch(
    *,
    batch: list,
    prompt_template: str,
) -> Batch:
    images = batch[0]
    captions = list(map(_decode_bytes_to_str, batch[1]))
    langs = list(map(_decode_bytes_to_str, batch[2]))
    sample_ids = list(map(_decode_bytes_to_str, batch[3]))
    keys = list(map(_decode_bytes_to_str, batch[4]))

    languages = [xflickrco_languages[lang] for lang in langs]
    prompts = [prompt_template.format(LANGUAGE=language) for language in languages]

    return _build_batch(
        prompts=prompts,
        gold_text=captions,
        images=images,
        languages=languages,
        sample_ids=sample_ids,
        keys=keys,
    )


def compute_scores(
    dataset: Literal[
        "marvl",
        "xgqa",
        "xm3600",
        "xvnli",
        "maxm",
        "xflickrco",
        "m5b_vgr",
        "m5b_vlod",
    ],
    preds_df: pd.DataFrame,
    lang: str,
    model_id: str,
    anno_file: str | None = None,
    use_gpu: bool = True,
):
    if dataset in {
        "marvl",
        "xgqa",
        "xvnli",
        "maxm",
        "m5b_vgr",
        "m5b_vlod",
    }:
        print(
            f"Computing generated label classification scores for {dataset} {lang} ...."
        )
        scores = generated_label_classification_evaluation(
            pred_labels=preds_df["pred_text"].tolist(),
            gold_labels=preds_df["gold_text"].tolist(),
            vqa_post_process=True,
            bool_to_yes_no=True,
            entailment_to_yes_no_maybe=dataset == "xvnli",
            remove_trailing_period=lang not in ["zh", "th"],
        )
    elif dataset in "xm3600":
        print(
            f"Computing generated caption generation scores for {dataset} {lang} ...."
        )
        if anno_file is None:
            raise ValueError(
                "You need to provide the annotation file for the XM3600 dataset!"
            )
        annos = srsly.read_json(anno_file)
        refs = {}
        for anno in annos["annotations"]:  # type: ignore
            aid = str(anno["image_id"])
            if aid in preds_df["sample_id"].values:
                if aid not in refs:  # type: ignore
                    refs[aid] = {"gold": []}  # type: ignore
                refs[aid]["gold"].append(anno["caption"])  # type: ignore

        preds_df.apply(
            lambda r: refs[r.sample_id].update({"pred": r.pred_text}), axis=1
        )

        pred_text = list(map(lambda r: r["pred"], refs.values()))
        gold_text = list(map(lambda r: r["gold"], refs.values()))
        scores = generated_caption_evaluation(
            caption_preds=pred_text,
            caption_golds=gold_text,
            lang_id=lang,
            use_gpu=use_gpu,
        )
    elif dataset in "xflickrco":
        print(
            f"Computing generated caption generation scores for {dataset} {lang} ...."
        )
        scores = generated_caption_evaluation(
            caption_preds=preds_df["pred_text"].tolist(),
            caption_golds=preds_df["gold_text"].tolist(),
            lang_id=lang,
            use_gpu=use_gpu,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

    scores = pd.DataFrame(scores, index=[0])
    scores.insert(0, "language", all_langs[lang])
    scores.insert(0, "lang_code", lang)
    scores.insert(0, "quant", "")
    scores.insert(0, "model_id", model_id)
    scores = scores.reset_index(drop=True)
    scores.sort_values("lang_code", inplace=True)

    print(f"Scores: {scores}")

    return scores


def _create_preds_df(
    gold_texts: List[str] | List[List[str]] | None = None,
    pred_texts: List[str] | None = None,
    keys: List[str] | None = None,
    sample_ids: List[str] | None = None,
    prompts: List[str] | None = None,
    languages: List[str] | None = None,
) -> pd.DataFrame:
    lang_preds_stream_df = pd.DataFrame(
        {
            "gold_text": [] if gold_texts is None else gold_texts,
            "pred_text": [] if pred_texts is None else pred_texts,
            "sample_id": [] if sample_ids is None else sample_ids,
            "prompt": [] if prompts is None else prompts,
            "language": [] if languages is None else languages,
            "key": [] if keys is None else keys,
        }
    )
    return lang_preds_stream_df


def _create_or_load_preds_df(
    preds_df_fn: Path,
) -> pd.DataFrame:
    if preds_df_fn.exists():
        try:
            preds_df = pd.read_csv(
                preds_df_fn,
                dtype={
                    "gold_text": str,
                    "pred_text": str,
                    "sample_id": str,
                    "prompt": str,
                    "language": str,
                    "key": str,
                },
            )
            assert all(
                c in preds_df.columns
                for c in [
                    "gold_text",
                    "pred_text",
                    "sample_id",
                    "prompt",
                    "language",
                    "key",
                ]
            ), "The columns in the predictions file are not as expected!"
        except pd.errors.EmptyDataError:
            return _create_preds_df()

        preds_df.drop_duplicates(subset=["key"], inplace=True)

        # when the gold text is a list, it is saved as a string in the csv (currently only for MaXM)
        if (
            "[" in preds_df["gold_text"].iloc[0]
            and "]" in preds_df["gold_text"].iloc[0]
        ):
            preds_df["gold_text"] = preds_df["gold_text"].apply(
                lambda x: x.strip("[]").replace("'", "").split(", ")
                if x != "[]"
                else list()
            )
        preds_df.to_csv(preds_df_fn, index=False)
    else:
        preds_df = _create_preds_df()
    return preds_df


def _recursive_move_to(
    item: Any, tgt: str | torch.dtype, criterion_func: Callable[..., bool]
):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [_recursive_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([_recursive_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: _recursive_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item


def _generate_pred_text_llava_liuhaotian(
    *,
    images: List[Image.Image],
    prompts: List[str],
    image_processor: ImageProcessor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM | LlavaForConditionalGeneration,
    model_id: str,
    model_dtype: torch.dtype,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    from llava.constants import (  # type: ignore
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates  # type: ignore
    from llava.mm_utils import (  # type: ignore
        process_images,
        tokenizer_image_token,
    )

    if "top_k" in gen_kwargs:
        del gen_kwargs["top_k"]

    pred_texts = []
    llava_prompts = []
    # we ignore the batch size here and just process one by one to avoid OOM
    for i, qs in enumerate(prompts):
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:  # type: ignore
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:  # type: ignore
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_id.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_id.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_id.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_id.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_id.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        llava_prompts.append(prompt)

        image_sizes = [images[i].size]
        images_tensor = process_images(
            [images[i]],
            image_processor,
            model.config,  # type: ignore
        ).to(
            model.device,  # type: ignore
            dtype=model_dtype,
        )

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        output_ids = model.generate(  # type: ignore
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=True,
            **gen_kwargs,  # type: ignore
        )

        pred_text = tokenizer.batch_decode(  # type: ignore
            output_ids,
            skip_special_tokens=True,
        )[0].strip()
        pred_texts.append(pred_text)
    return pred_texts


def __image_to_b64(image: Image.Image) -> str:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    img_bytes = image_bytes.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def _generate_pred_text_omni_lmm(
    *,
    images: List[Image.Image],
    prompts: List[str],
    model,
) -> list[str]:
    pred_texts = []
    from omnilmm_chat import OmniLMMChat

    model = cast(OmniLMMChat, model)
    base64_images = [__image_to_b64(image) for image in images]

    # we ignore the batch size here and just process one by one because it's straightforward
    for b64_img, prompt in zip(base64_images, prompts):
        msgs = [{"role": "user", "content": prompt}]
        inputs = {"image": b64_img, "question": srsly.json_dumps(msgs)}

        pred_text = model.chat(inputs).strip()
        pred_texts.append(pred_text)
    return pred_texts


def _generate_pred_text_mini_cpm_v(
    *,
    images: List[Image.Image],
    prompts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    pred_texts = []

    # we ignore the batch size here and just process one by one because it's straightforward
    for img, prompt in zip(images, prompts):
        msgs = [{"role": "user", "content": prompt}]
        pred_text, _, _ = model.chat(  # type: ignore
            image=img,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=gen_kwargs["do_sample"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )

        pred_texts.append(pred_text)
    return pred_texts


def _generate_pred_text_qwen_vl_chat(
    *,
    images: List[Image.Image],
    prompts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    pred_texts = []

    # we ignore the batch size here and just process one by one because it's straightforward
    for img, prompt in zip(images, prompts):
        with NamedTemporaryFile(suffix=".jpg") as temp_file:
            img.save(temp_file.name)
            query = tokenizer.from_list_format(  # type: ignore
                [
                    {"image": temp_file.name},
                    {"text": prompt},
                ]
            )
            pred_text, _ = model.chat(  # type: ignore
                tokenizer,
                query=query,
                history=None,  # , **gen_kwargs
            )
        pred_texts.append(pred_text)
    return pred_texts


def _generate_pred_text_yi_vl(
    *,
    images: List[Image.Image],
    prompts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    image_processor: ImageProcessor,
    device: str,
    model_dtype: torch.dtype,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    pred_texts = []

    from yi.llava.conversation import conv_templates  # type: ignore
    from yi.llava.mm_utils import (  # type: ignore
        KeywordsStoppingCriteria,
        expand2square,
        tokenizer_image_token,
    )
    from yi.llava.model.constants import (  # type: ignore
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )

    # we ignore the batch size here and just process one by one because it's straightforward
    for prompt, image in zip(prompts, images):
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["mm_default"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        if getattr(model.config, "image_aspect_ratio", None) == "pad":  # type: ignore
            image = expand2square(
                image,
                tuple(int(x * 255) for x in image_processor.image_mean),  # type: ignore
            )
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[  # type: ignore
            "pixel_values"
        ][0]

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids = model.generate(  # type: ignore
            input_ids,
            images=image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype),
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **gen_kwargs,
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(  # type: ignore
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        pred_text = outputs.strip()
        pred_texts.append(pred_text)
    return pred_texts


def _generate_pred_text_llava_hf(
    *,
    images: List[Image.Image],
    prompts: List[str],
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    device: str,
    model_dtype: torch.dtype,
    model: AutoModelForCausalLM | LlavaForConditionalGeneration,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    pixel_values = image_processor(images)  # type: ignore
    prompt_prepro = text_processor(prompts)  # type: ignore

    model_inputs = {
        "input_ids": prompt_prepro.input_ids,
        "attention_mask": prompt_prepro.attention_mask,
        "pixel_values": pixel_values,
    }
    model_inputs = _recursive_move_to(
        model_inputs, device, lambda x: isinstance(x, torch.Tensor)
    )
    model_inputs = _recursive_move_to(
        model_inputs,
        model_dtype,
        lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
    )
    output_ids = model.generate(**model_inputs, **gen_kwargs)  # type: ignore
    output_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]  # type: ignore
    pred_text = text_processor.batch_decode(output_ids, skip_special_tokens=True)

    return pred_text


def _generate_pred_text_cogvlm_hf(
    *,
    images: List[Image.Image],
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: LlamaTokenizer,
    device: str,
    model_dtype: torch.dtype,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    features = [
        model.build_conversation_input_ids(  # type: ignore
            tokenizer,
            query=prompt,
            history=[],
            images=[img],
        )
        for img, prompt in zip(images, prompts)
    ]
    images = [feature.pop("images") for feature in features]
    tokenizer.padding_side = "left"
    padded_features = tokenizer.pad(features)

    model_inputs = {
        **padded_features,
        "images": images,
    }
    model_inputs = _recursive_move_to(
        model_inputs, device, lambda x: isinstance(x, torch.Tensor)
    )
    model_inputs = _recursive_move_to(
        model_inputs,
        model_dtype,
        lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
    )
    output_ids = model.generate(**model_inputs, **gen_kwargs)  # type: ignore
    output_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]  # type: ignore
    pred_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return pred_text


def _generate_pred_text_mblip_hf(
    *,
    images: List[Image.Image],
    prompts: List[str],
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    device: str,
    model_dtype: torch.dtype,
    gen_kwargs: Dict[str, str | int | float],
) -> List[str]:
    inputs = processor(images, prompts, padding=True, return_tensors="pt")

    model_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values,
    }

    model_inputs = _recursive_move_to(
        model_inputs, device, lambda x: isinstance(x, torch.Tensor)
    )
    model_inputs = _recursive_move_to(
        model_inputs,
        model_dtype,
        lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
    )
    output_ids = model.generate(**model_inputs, **gen_kwargs)  # type: ignore

    # mblip does not return the prompt, so we need to remove the respective tokens
    pred_text = processor.batch_decode(output_ids, skip_special_tokens=True)

    return pred_text


def _generate_pred_text_api_model(
    *,
    images: List[
        Image.Image
    ],  # TODO support multiple images per prompt (for MaRVL, VGR, VLOD)
    prompts: List[str],
    model: Union[
        "GeminiProVisionGoogleStudioModel",
        "GeminiProVisionVertexAIModel",
        "GPT4VisionOpenAIModel",
    ],
    gen_kwargs: Dict[str, str | int | float],
) -> List[str]:
    pred_texts = []
    for prompt, imgs in zip(prompts, images):
        pred_text = model.generate_content(
            prompt,
            imgs,
            max_new_tokens=int(gen_kwargs["max_new_tokens"]),
            temperature=float(gen_kwargs["temperature"]),
            top_k=int(gen_kwargs["top_k"]),
            top_p=int(gen_kwargs["top_p"]),
        )
        pred_texts.append(pred_text)
    return pred_texts


def _generate_pred_text_intern_vl(
    *,
    images: List[Image.Image],
    prompts: List[str],
    image_processor: ImageProcessor,
    tokenizer: AutoTokenizer,
    device: str,
    model_dtype: torch.dtype,
    model: AutoModelForCausalLM,
    gen_kwargs: Dict[str, str | int | float],
) -> list[str]:
    if gen_kwargs["num_beams"] > 1:
        print("num_beams > 1 not supported for Intern-VL models!")
    if gen_kwargs["do_sample"]:
        print("do_sample=True not supported for Intern-VL models!")

    gen_kwargs["do_sample"] = False
    gen_kwargs["num_beams"] = 1
    if "top_p" in gen_kwargs:
        del gen_kwargs["top_p"]

    # we ignore the batch size here and just process one by one because it's straightforward
    pred_texts = []
    for prompt, image in zip(prompts, images):
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model_dtype).cuda()

        pred_text = model.chat(tokenizer, pixel_values, prompt, gen_kwargs)
        pred_texts.append(pred_text)

    return pred_texts


def _filter_batch(
    batch: Batch,
    lang_preds_stream_df: pd.DataFrame,
) -> Batch:
    # Skip samples that are already present in lang_preds_stream_df
    filtered_batch = _build_batch(
        prompts=[],
        gold_text=[],
        images=[],
        languages=[],
        sample_ids=[],
        keys=[],
    )
    for sample_idx in range(len(batch["sample_ids"])):
        key = batch["keys"][sample_idx]
        if key not in lang_preds_stream_df["key"].values:
            for f in Batch.fields():
                filtered_batch[f].append(batch[f][sample_idx])
    return filtered_batch


def generate_pred_text(
    *,
    model: Union[
        AutoModelForCausalLM,
        LlavaForConditionalGeneration,
        Blip2ForConditionalGeneration,
        "GeminiProVisionGoogleStudioModel",
        "GeminiProVisionVertexAIModel",
        "GPT4VisionOpenAIModel",
    ],
    tokenizer: AutoTokenizer | LlamaTokenizer | None,
    image_processor: ImageProcessor | None,
    text_processor: TextProcessor | None,
    processor: Blip2Processor | None,
    model_id: str,
    device: str,
    model_dtype: torch.dtype,
    batch: Batch,
    lang_preds_stream_df: pd.DataFrame,
    gen_kwargs: Dict[str, str | int | float],
):
    batch = _filter_batch(batch, lang_preds_stream_df)

    if len(batch["sample_ids"]) == 0:
        return lang_preds_stream_df

    if model_id.startswith("llava-hf/"):
        pred_text = _generate_pred_text_llava_hf(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            image_processor=image_processor,
            text_processor=text_processor,
            device=device,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("THUDM/cogvlm-chat-hf"):
        pred_text = _generate_pred_text_cogvlm_hf(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("Gregor/mblip-"):
        pred_text = _generate_pred_text_mblip_hf(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            processor=processor,
            device=device,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("liuhaotian/llava-"):
        pred_text = _generate_pred_text_llava_liuhaotian(
            images=batch["images"],
            prompts=batch["prompts"],
            image_processor=image_processor,
            tokenizer=tokenizer,
            model=model,
            model_id=model_id,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("openbmb/OmniLMM-"):
        pred_text = _generate_pred_text_omni_lmm(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
        )
    elif model_id.startswith("openbmb/MiniCPM-V"):
        pred_text = _generate_pred_text_mini_cpm_v(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("Qwen/Qwen-VL-Chat"):
        pred_text = _generate_pred_text_qwen_vl_chat(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("01-ai/Yi-VL-"):
        pred_text = _generate_pred_text_yi_vl(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("OpenGVLab/InternVL-Chat-"):
        pred_text = _generate_pred_text_intern_vl(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            model_dtype=model_dtype,
            gen_kwargs=gen_kwargs,
        )
    elif model_id.startswith("gpt-4") or model_id.startswith("gemini-pro-vision"):
        pred_text = _generate_pred_text_api_model(
            images=batch["images"],
            prompts=batch["prompts"],
            model=model,
            gen_kwargs=gen_kwargs,
        )
    else:
        raise ValueError(f"Model {model_id} not supported!")

    pred_texts: List[str] = pred_text
    gold_texts: List[str] = batch["gold_text"]  # type: ignore
    keys: List[str] = batch["keys"]  # type: ignore
    sample_ids: List[str] = batch["sample_ids"]  # type: ignore
    prompts: List[str] = batch["prompts"]  # type: ignore
    languages: List[str] = batch["languages"]  # type: ignore

    lang_preds_stream_df = pd.concat(
        [
            lang_preds_stream_df,
            _create_preds_df(
                gold_texts,
                pred_texts,
                keys,
                sample_ids,
                prompts,
                languages,
            ),
        ]
    )

    return lang_preds_stream_df


def _get_torch_dtype(dtype: str) -> torch.dtype:
    str2dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    return str2dtype[dtype]


def _load_mblip(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    Blip2ForConditionalGeneration,
    AutoTokenizer | LlamaTokenizer | None,
    ImageProcessor | None,
    TextProcessor | None,
    Blip2Processor | None,
]:
    if flash_attn:
        print("Flash attention is not supported for MBlip models! Ignoring ...")

    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if load_8bit or load_4bit:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
        }

    model = Blip2ForConditionalGeneration.from_pretrained(
        hf_model_id,
        **model_kwargs,
    )
    model = cast(Blip2ForConditionalGeneration, model)

    text_processor = None
    image_processor = None
    tokenizer = None
    processor = Blip2Processor.from_pretrained(hf_model_id)

    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)  # type: ignore
    model.eval()

    return model, tokenizer, image_processor, text_processor, processor


def _load_llava_hf(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    LlavaForConditionalGeneration,
    None,
    ImageProcessor | None,
    TextProcessor | None,
    None,
]:
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if load_8bit or load_4bit:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
        }
    model_kwargs["attn_implementation"] = "flash_attention_2" if flash_attn else "sdpa"
    # TODO https://github.com/haotian-liu/LLaVA
    print(f"Loading Huggingface LLaVA model: {hf_model_id} in {torch_dtype} ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id,
        **model_kwargs,
    )
    model = cast(LlavaForConditionalGeneration, model)
    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)  # type: ignore
    model.eval()

    image_processor = ImageProcessor(hf_model_id)
    text_processor = TextProcessor(hf_model_id)
    tokenizer = None
    processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_cogvlm(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModelForCausalLM,
    LlamaTokenizer,
    None,
    None,
    None,
]:
    if flash_attn:
        print("Flash attention is not supported for CogVLM models! Ignoring ...")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if load_8bit or load_4bit:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
        }

    print(f"Loading THUDM CogVLM model: {hf_model_id} in {torch_dtype} ...")
    model_kwargs["trust_remote_code"] = True
    if load_8bit or load_4bit:
        print("Quantization is not supported for THUDM CogVLM models! Ignoring ...")
        del model_kwargs["device_map"]
        del model_kwargs["load_in_4bit"]
        del model_kwargs["load_in_8bit"]
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/cogvlm-chat-hf",
        **model_kwargs,
    )

    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)  # type: ignore
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    image_processor = None
    text_processor = None
    processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_liuhaotian_llava(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModelForCausalLM,
    Any,
    None,
    None,
    None,
]:
    print(f"Loading Liu Haotian LLaVA model: {hf_model_id} in {torch_dtype} ...")

    from llava.mm_utils import get_model_name_from_path  # type: ignore
    from llava.model.builder import load_pretrained_model  # type: ignore

    model_name = get_model_name_from_path(hf_model_id)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        hf_model_id,
        model_base,
        model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map="auto",
        device=device,
        use_flash_attn=flash_attn,
    )
    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)  # type: ignore
    model.eval()
    text_processor = None
    processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_omni_lmm(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModelForCausalLM,
    None,
    None,
    None,
    None,
]:
    print(f"Loading OmniLMM model: {hf_model_id} in {torch_dtype} ...")

    from omnilmm_chat import OmniLMMChat

    model = OmniLMMChat(hf_model_id)
    tokenizer = image_processor = text_processor = processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_mini_cpm_v(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModel,
    AutoTokenizer,
    None,
    None,
    None,
]:
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if load_8bit or load_4bit:
        model_kwargs = {
            "device_map": "auto",
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
        }
    model_kwargs["attn_implementation"] = "flash_attention_2" if flash_attn else "sdpa"

    print(f"Loading MiniCPM-V model: {hf_model_id} in {torch_dtype} ...")

    model = AutoModel.from_pretrained(
        hf_model_id,
        **model_kwargs,
    )
    model = cast(AutoModel, model)
    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)  # type: ignore
    model.eval()  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)

    image_processor = text_processor = processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_qwen_vl(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModel,
    AutoTokenizer,
    None,
    None,
    None,
]:
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_8bit or load_4bit:
        model_kwargs = {
            "device_map": "auto",
            "load_in_4bit": load_4bit,
            "load_in_8bit": load_8bit,
        }

    if flash_attn:
        print("Flash attention is not supported for QwenVL models! Ignoring ...")

    print(f"Loading Qwen-VL model: {hf_model_id} in {torch_dtype} ...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        **model_kwargs,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        hf_model_id, trust_remote_code=True
    )
    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=torch_dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)

    image_processor = text_processor = processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_yi_vl(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModel,
    AutoTokenizer,
    ImageProcessor,
    None,
    None,
]:
    if load_8bit or load_4bit:
        print("Quantization is not supported for Yi VL models! Ignoring ...")

    if not flash_attn:
        print("Flash attention is mandatory for Yi VL models! Ignoring ...")

    print(f"Loading Yi VL model: {hf_model_id} in {torch_dtype} ...")

    from yi.llava.mm_utils import (  # type: ignore
        get_model_name_from_path,
        load_pretrained_model,
    )
    from yi.llava.model.constants import (  # type: ignore
        key_info,
    )

    disable_torch_init()
    model_path = Path(os.getcwd()) / "yi" / hf_model_id
    if not model_path.exists():
        raise ValueError(
            f"Model path {model_path} does not exist! Make sure to download the model correctly!"
        )
    model_path = str(model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, device_map=device
    )

    text_processor = processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_intern_vl(
    hf_model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
) -> Tuple[
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    None,
    None,
]:
    if load_8bit and load_4bit:
        raise ValueError("Cannot load model in both 4bit and 8bit!")

    if flash_attn:
        print("Flash attention is not supported by Intern VL models! Ignoring ...")

    print(f"Loading Intern VL model: {hf_model_id} in {torch_dtype} ...")

    model = AutoModel.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.device_count() > 1 else device,
        load_in_4bit=load_4bit,
        load_in_8bit=load_8bit,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    image_processor = CLIPImageProcessor.from_pretrained(hf_model_id)
    processor = text_processor = None

    return model, tokenizer, image_processor, text_processor, processor


def _load_api_model(
    model_id: str,
    api_model_key: str | None,
    google_project_id: str | None,
    gpt_4_img_detail: Literal["low", "auto", "high"],
) -> Tuple[
    Union[
        "GeminiProVisionGoogleStudioModel",
        "GeminiProVisionVertexAIModel",
        "GPT4VisionOpenAIModel",
    ],
    None,
    None,
    None,
    None,
]:
    from m5b.model.api_models import load_api_model

    model = load_api_model(
        model_id=model_id,
        api_model_key=api_model_key,
        google_project_id=google_project_id,
        gpt_4_img_detail=gpt_4_img_detail,
    )
    tokenizer = None
    image_processor = None
    text_processor = None
    processor = None
    return model, tokenizer, image_processor, text_processor, processor


def load_model(
    model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    gpt_4_img_detail: Literal["low", "auto", "high"] = "low",
    api_model_key: str | None = None,
    google_project_id: str | None = None,
) -> Tuple[
    Union[
        LlavaForConditionalGeneration,
        Blip2ForConditionalGeneration,
        AutoModelForCausalLM,
        AutoModel,
        "GeminiProVisionGoogleStudioModel",
        "GeminiProVisionVertexAIModel",
        "GPT4VisionOpenAIModel",
    ],
    AutoTokenizer | LlamaTokenizer | None,
    ImageProcessor | CLIPImageProcessor | None,
    TextProcessor | None,
    Blip2Processor | None,
]:
    if load_8bit or load_4bit:
        torch_dtype = torch.bfloat16

    if load_8bit and load_4bit:
        raise ValueError("Cannot load model in both 4bit and 8bit!")

    if model_id.startswith("llava-hf/"):
        return _load_llava_hf(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("Gregor/mblip-"):
        return _load_mblip(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("THUDM/cogvlm-chat-hf"):
        return _load_cogvlm(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )

    elif model_id.startswith("liuhaotian/llava-"):
        return _load_liuhaotian_llava(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("openbmb/OmniLMM-12B"):
        return _load_omni_lmm(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("openbmb/MiniCPM-V"):
        return _load_mini_cpm_v(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("Qwen/Qwen-VL-Chat"):
        return _load_qwen_vl(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("01-ai/Yi-VL-"):
        return _load_yi_vl(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("OpenGVLab/InternVL-Chat-"):
        return _load_intern_vl(
            model_id,
            torch_dtype,
            device,
            load_8bit,
            load_4bit,
            flash_attn,
        )
    elif model_id.startswith("gpt-4") or model_id.startswith("gemini-"):
        return _load_api_model(
            model_id,
            api_model_key,
            google_project_id,
            gpt_4_img_detail,
        )
    else:
        raise NotImplementedError(f"Model ID {model_id} not supported!")


def main(
    dataset: Literal[
        "marvl",
        "xgqa",
        "xm3600",
        "xvnli",
        "maxm",
        "xflickrco",
        "m5b_vgr",
        "m5b_vlod",
    ],
    model_id: str,
    data_base_path: str | Path,
    log_dir: str | Path | None = None,
    max_samples_per_lang: int | None = None,
    prompt_template: str | None = None,
    langs: List[str] | None = None,
    marvl_use_english_translation: Literal[
        "en-tr", "en-id", "en-ta", "en-zh", "en-sw", "all"
    ]
    | None = None,
    marvl_only_use_translation: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bfloat16",
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    batch_size: int = 4,
    num_workers: int = 8,
    seed: int = 1337,
    # for multi image datasets (e.g., MaRVL)
    use_stacked_images: Literal[
        "vertically", "horizontally", "two_cols"
    ] = "horizontally",
    # Generation kwargs
    num_beams: int = 5,
    do_sample: bool = True,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    # only run eval (no pred)
    only_eval: bool = False,
    # for api models
    api_model_key: str | None = None,
    google_project_id: str | None = None,
    gpt_4_img_detail: Literal["low", "auto", "high"] = "low",
    # disable gpu for eval
    use_gpu_for_eval: bool = True,
):
    """
    Main function for evaluating a model on a dataset.

    Args:
        dataset: Dataset to evaluate on.
        model_id: Model ID to evaluate.
        data_base_path: Base path to the datasets. The dataset will be loaded from data_base_path/dataset.
        log_dir: Directory to store logs and results. If None, the log_dir is set to $PWD/logs/eval_{dataset}.
        max_samples_per_lang: Maximum number of samples per language to evaluate on. If None, all samples are evaluated. Default is None.
        prompt_template: Prompt template to use for the dataset. If None, the default prompt template for the dataset is used. Default is None.
        langs: Languages to evaluate on. If None, all languages are evaluated. Default is None.
        marvl_use_english_translation: Use English translations for MaRVL. Default is None.
        marvl_only_use_translation: Only use translations for MaRVL. Default is False.
        device: Device to use for evaluation. Default is "cuda" if available, else "cpu".
        dtype: Data type to use for evaluation. Default is "bfloat16".
        load_8bit: Load model in 8bit. Default is False.
        load_4bit: Load model in 4bit. Default is False.
        flash_attn: Use Flash Attention for the model. Default is False.
        batch_size: Batch size for evaluation. Default is 4.
        num_workers: Number of workers for DataLoader. Default is 8.
        seed: Seed for random number generators. Default is 1337.
        use_stacked_images: How to stack images for multi image datasets. Default is "horizontally".
        num_beams: Number of beams for generation. Default is 5.
        do_sample: Use sampling for generation. Default is True.
        max_new_tokens: Maximum number of new tokens to generate. Default is 50.
        temperature: Temperature for generation. Default is 1.0.
        top_k: Top K for generation. Default is 50.
        top_p: Top P for generation. Default is 0.95.
        only_eval: Only run evaluation (no prediction). Default is False.
        api_model_key: API model key for API models. Default is None.
        google_project_id: Google project ID for API models. Default is None.
        gpt_4_img_detail: Image detail for GPT-4 Vision API models. Default is "low".
        use_gpu_for_eval: Use GPU for evaluation. Default is True.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(seed)

    if dataset not in {
        "marvl",
        "xgqa",
        "xm3600",
        "xvnli",
        "maxm",
        "xflickrco",
        "m5b_vgr",
        "m5b_vlod",
    }:
        raise ValueError(f"Dataset {dataset} not supported!")

    prompt_template = _check_or_create_prompt_template(
        dataset, model_id, prompt_template
    )

    if data_base_path is None:
        raise ValueError("Data base path is required!")
    elif isinstance(data_base_path, (Path, str)):
        data_base_path = Path(data_base_path) / f"{dataset}"
    if dataset not in str(data_base_path):
        raise ValueError(
            f"Data base path {data_base_path} does not match dataset {dataset}!"
        )

    if log_dir is None:
        log_dir = Path(".") / f"logs/eval_{dataset}"
    else:
        log_dir = Path(log_dir)
    if dataset not in str(log_dir):
        raise ValueError(f"Log dir {log_dir} does not match dataset {dataset}!")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    torch_dtype = _get_torch_dtype(dtype)
    if load_8bit or load_4bit:
        torch_dtype = torch.bfloat16

    gen_kwargs = {
        "num_beams": num_beams,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    print(f"Using generation kwargs: {gen_kwargs}")

    all_lang_scores_df = []
    model_name = model_id.split("/")[-1]
    quant_infix = "_8bit" if load_8bit else "_4bit" if load_4bit else ""
    stacked_imgs_infix = (
        f"_{use_stacked_images}" if dataset in ["marvl", "m5b_vlod", "m5b_vgr"] else ""
    )

    if only_eval:
        model, tokenizer, image_processor, text_processor, processor = (
            None,
            None,
            None,
            None,
            None,
        )
    else:
        model, tokenizer, image_processor, text_processor, processor = load_model(
            model_id=model_id,
            torch_dtype=torch_dtype,
            device=device,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            flash_attn=flash_attn,
            gpt_4_img_detail=gpt_4_img_detail,
            api_model_key=api_model_key,
            google_project_id=google_project_id,
        )

    ds = load_dataset(
        dataset=dataset,
        data_base_path=data_base_path,
        langs=langs,
        prompt_template=prompt_template,
        batch_size=batch_size,
        num_workers=num_workers,
        use_stacked_images=use_stacked_images,
        marvl_use_english_translation=marvl_use_english_translation,
        marvl_only_use_translation=marvl_only_use_translation,
    )

    desc = "Evaluating language: {LANG}"

    with torch.inference_mode():
        for dl_idx, lang_dl in (
            pbar := tqdm(
                enumerate(ds.test_dataloader()),
                total=len(ds.languages),
                position=0,
                leave=True,
                desc=desc,
            )
        ):
            lang = ds.get_lang_id(dl_idx)

            all_lang_preds_fn = (
                log_dir
                / f"{model_name}{quant_infix}{stacked_imgs_infix}_preds_{lang}.csv"
            )
            preds_stream_fn = (
                log_dir
                / f"{model_name}{quant_infix}{stacked_imgs_infix}_preds_{lang}_stream.csv"
            )
            lang_scores_fn = (
                log_dir
                / f"{model_name}{quant_infix}{stacked_imgs_infix}_scores_{lang}.csv"
            )
            all_scores_fn = (
                log_dir / f"{model_name}{quant_infix}{stacked_imgs_infix}_scores.csv"
            )

            print(f"{all_lang_preds_fn=}")
            print(f"{preds_stream_fn=}")
            print(f"{lang_scores_fn=}")
            print(f"{all_scores_fn=}")

            pbar.set_description(desc.format(LANG=lang))
            if all_lang_preds_fn.exists():
                all_lang_preds_df = _create_or_load_preds_df(all_lang_preds_fn)
                print(
                    f"All results for {lang} already exist at {all_lang_preds_fn} with {len(all_lang_preds_df)} samples!"
                    "If you want to re-run, delete the file."
                )
            elif only_eval and not all_lang_preds_fn.exists():
                raise ValueError(
                    f"Only eval is set to True, but {all_lang_preds_fn} does not exist!"
                )
            else:
                lang_preds_stream_df = _create_or_load_preds_df(preds_stream_fn)
                if len(lang_preds_stream_df) > 0:
                    print(
                        f"Partial results for {lang} already exist at {preds_stream_fn} with {len(lang_preds_stream_df)} entries! Continuing..."
                    )
                for batch in tqdm(
                    lang_dl, position=1, leave=False, desc="Processing batches..."
                ):
                    match dataset:
                        case "marvl":
                            batch = _marvl_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                                use_stacked_images=use_stacked_images,
                                translation=marvl_use_english_translation is not None,
                            )
                        case "xgqa":
                            batch = _xgqa_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                            )
                        case "xm3600":
                            batch = _xm3600_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                            )
                        case "xvnli":
                            batch = _xvnli_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                            )
                        case "maxm":
                            batch = _maxm_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                            )
                        case "xflickrco":
                            batch = _xflickrco_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                            )
                        case "m5b_vgr":
                            batch = _m5b_vgr_build_batch(
                                batch=batch,
                                prompt_template=prompt_template,
                                use_stacked_images=use_stacked_images,
                            )
                        case "m5b_vlod":
                            batch = _m5b_vlod_build_batch(
                                batch=batch,
                                use_stacked_images=use_stacked_images,
                                choices_type=ds.choices_type,
                                prompt_template=ds.build_preliminary_prompt_template(
                                    prompt_template
                                ),
                            )
                        case _:
                            raise ValueError(f"Dataset {dataset} not supported!")

                    lang_preds_stream_df = generate_pred_text(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        text_processor=text_processor,
                        processor=processor,
                        model_id=model_id,
                        device=device,
                        model_dtype=torch_dtype,
                        batch=batch,
                        lang_preds_stream_df=lang_preds_stream_df,
                        gen_kwargs=gen_kwargs,
                    )

                    lang_preds_stream_df.to_csv(preds_stream_fn, index=False)
                    if (
                        max_samples_per_lang is not None
                        and len(lang_preds_stream_df) >= max_samples_per_lang
                    ):
                        print(
                            f"Stopping after {max_samples_per_lang} samples for {lang}!"
                        )
                        break

                print(
                    f"Saving all {len(lang_preds_stream_df)} predictions for {lang} to {all_lang_preds_fn} ...."
                )
                all_lang_preds_df = lang_preds_stream_df
                all_lang_preds_df.to_csv(all_lang_preds_fn, index=False)

            lang_scores_df = compute_scores(
                dataset=dataset,
                preds_df=all_lang_preds_df,
                lang=lang,
                model_id=model_id,
                anno_file=None
                if dataset != "xm3600"
                else ds.get_language_coco_json_file(lang),  # type: ignore
                use_gpu=use_gpu_for_eval,
            )
            print(f"Saving scores for {lang} to {lang_scores_fn} ....")
            lang_scores_df.to_csv(lang_scores_fn, index=False)
            all_lang_scores_df.append(lang_scores_df)

    all_lang_scores_df = pd.concat(all_lang_scores_df)
    print("All done! All scores:")
    print(all_lang_scores_df)
    print(f"Saving all scores to {all_scores_fn} ....")
    all_lang_scores_df.to_csv(all_scores_fn, index=False)


if __name__ == "__main__":
    fire.Fire(main)
