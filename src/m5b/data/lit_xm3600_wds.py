import json
from pathlib import Path
from typing import Dict, List

import webdataset as wds
from lightning import LightningDataModule
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlavaForConditionalGeneration,
)

from m5b.data.image_processor import ImageProcessor
from m5b.data.text_processor import TextProcessor
from m5b.util.caption_prompter import CaptionPrompter
from m5b.util.data import build_wds, build_wds_dataloader
from m5b.util.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LitXM3600WDS(LightningDataModule):
    def __init__(
        self,
        data_base_path: str | Path,
        text_processor: TextProcessor | None = None,
        image_processor: ImageProcessor | None = None,
        model: AutoModelForCausalLM | LlavaForConditionalGeneration | None = None,
        tokenizer: LlamaTokenizer | None = None,
        caption_prompter: CaptionPrompter | None = None,
        batch_size: int = 32,
        num_workers: int = 16,
        languages: Dict[str, str] = {
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
        },
        no_collate: bool = False,
        skip_duplicate_images: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_base_path = Path(data_base_path)
        self.languages = languages
        self.idx2lang = list(languages.keys())
        self._coco_json_files: Dict[str, str] = dict()
        self._dataloaders: Dict[str, wds.WebLoader] = dict()
        self._dataloader_sizes: Dict[str, int] = dict()

        self.caption_prompter = caption_prompter
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.model = model
        self.tokenizer = tokenizer
        self.no_collate = no_collate
        self._use_cogvlm_input_preprocessing = False

        if not self.no_collate:
            if image_processor is not None and text_processor is not None:
                if not (
                    isinstance(self.image_processor, ImageProcessor)
                    or callable(self.image_processor)
                ):
                    raise ValueError(
                        "image_processor is not a ImageProcessor nor a callable"
                    )
                if not (
                    isinstance(self.text_processor, TextProcessor)
                    or callable(self.text_processor)
                ):
                    raise ValueError(
                        "text_processor is not a TextProcessor nor a callable"
                    )
            elif model is not None and tokenizer is not None:
                if not callable(model.build_conversation_input_ids):
                    # for CogVLM... This is ugly, I know :( Probably better provide a collate function instead. But this does not work well with hydra
                    raise ValueError(
                        "model must have a build_conversation_input_ids method"
                    )
                self._use_cogvlm_input_preprocessing = True
            elif model is None or tokenizer is None:
                raise ValueError(
                    "Both model and tokenizer must be provided if one of them is provided"
                )
            else:
                raise ValueError(
                    "Either model and tokenizer or image_processor and text_processor must be provided, not both."
                )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.no_collate = no_collate
        self.skip_duplicate_images = skip_duplicate_images if batch_size > 1 else False

    def prepare_data(self):
        assert (
            self.data_base_path.exists()
        ), f"data_base_path does not exist: {self.data_base_path}"

        for lang in self.languages:
            wds_base_path = self.data_base_path / "wds" / lang
            assert (
                wds_base_path.exists()
            ), f"wds path for langauge '{lang}' does not exist: {wds_base_path}"

            anno_file = self.data_base_path / f"xm3600_coco_{lang}.json"
            assert (
                anno_file.exists() and anno_file.is_file()
            ), f"Annotation file for language '{lang}' does not exist: {anno_file}"
            self._coco_json_files[lang] = str(anno_file)

            with open(anno_file, "r") as fp:
                data = json.load(fp)
                self._dataloader_sizes[lang] = len(
                    data["annotations"]
                )  # // self.batch_size

            self._dataloaders[lang] = self._build_dataloader_for_lang(lang)
            log.info(
                f"Built XM3600 WDS dataloader for language '{lang}'! with {self._dataloader_sizes[lang]} samples!"
            )

    def _build_dataloader_for_lang(self, lang: str) -> wds.WebLoader:
        wds = build_wds(
            wds_path=self.data_base_path / "wds" / lang,
            decode="pil",
            tuple_content=("jpg;png", "txt", "lang", "image_id", "__key__"),
            batch_size=self.batch_size,
            ds_size=self._dataloader_sizes[lang],
            map_tuple=None,
            shuffle=False,
        )

        def _collate_fn(batch):
            # TODO move this to the build_wds function
            def _decode(x: str | bytes) -> str:
                if isinstance(x, (bytes, bytearray)):
                    return x.decode("utf-8")
                else:
                    return x  # type: ignore

            images = batch[0]
            captions = list(map(_decode, batch[1]))
            langs = list(map(_decode, batch[2]))
            image_ids = list(map(_decode, batch[3]))
            keys = list(map(_decode, batch[4]))

            if self.skip_duplicate_images:
                # skip duplicate images based on image_id...
                # Since the images are exactly the same, we can just skip the duplicates
                # because they will lead to the same captions or outputs in general
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
                images = unique_images
                captions = unique_captions
                langs = unique_langs
                image_ids = unique_image_ids
                keys = unique_keys

            languages = [self.languages[lang] for lang in langs]
            prompts = self.caption_prompter.generate_prompts(langs=languages)  # type: ignore

            if self._use_cogvlm_input_preprocessing:
                features = [
                    self.model.build_conversation_input_ids(
                        self.tokenizer,
                        query=prompt,
                        history=[],
                        images=[img],
                    )
                    for img, prompt in zip(images, prompts)
                ]
                images = [feature.pop("images") for feature in features]
                self.tokenizer.padding_side = "left"
                padded_features = self.tokenizer.pad(features)

                return {
                    **padded_features,
                    "images": images,
                    "languages": languages,
                    "gold_text": captions,
                    "prompts": prompts,
                    "image_ids": image_ids,
                    "keys": keys,
                }
            else:
                pixel_values = self.image_processor(images)  # type: ignore
                prompt_prepro = self.text_processor(prompts)  # type: ignore

                return {
                    "input_ids": prompt_prepro.input_ids,  # B x S
                    "attention_mask": prompt_prepro.attention_mask,  # B x S
                    "pixel_values": pixel_values,  # B x C x H x W
                    "languages": languages,
                    "gold_text": captions,
                    "prompts": prompts,
                    "image_ids": image_ids,
                    "keys": keys,
                }

        return build_wds_dataloader(
            wds,
            num_workers=self.num_workers,
            collate_fn=_collate_fn if not self.no_collate else None,
        )

    def get_lang_id(self, lang_idx: int) -> str:
        return self.idx2lang[lang_idx]

    def get_language_dataloader(self, lang_id: str) -> wds.WebLoader:
        return self._dataloaders[lang_id]

    def get_language_coco_json_file(self, lang: str) -> str:
        return self._coco_json_files[lang]

    def train_dataloader(self):
        raise NotImplementedError("XM3600 is only meant for evaluation!")

    def val_dataloader(self):
        raise NotImplementedError("XM3600 is only meant for evaluation!")

    def test_dataloader(self) -> List[wds.WebLoader]:
        return list(self._dataloaders.values())
