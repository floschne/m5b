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
from m5b.util.data import build_wds, build_wds_dataloader
from m5b.util.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LitXVNLIWDS(LightningDataModule):
    def __init__(
        self,
        data_base_path: str | Path,
        text_processor: TextProcessor | None = None,
        image_processor: ImageProcessor | None = None,
        model: AutoModelForCausalLM | LlavaForConditionalGeneration | None = None,
        tokenizer: LlamaTokenizer | None = None,
        batch_size: int = 32,
        num_workers: int = 16,
        languages: Dict[str, str] = {
            "ar": "Arabic",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "ru": "Russian",
        },
        prompt_template: str = "Is it guaranteed true that ”{HYPOTHESIS}”? Yes, no, or maybe? One word answer in English:",
        no_collate: bool = False,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.data_base_path = Path(data_base_path)
        self.languages = languages
        self.idx2lang: Dict[int, str] = dict()
        self._dataloaders: Dict[str, wds.WebLoader] = dict()

        self.prompt_template = prompt_template
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.model = model
        self.tokenizer = tokenizer
        self._use_cogvlm_input_preprocessing = False
        self.no_collate = no_collate
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

    def prepare_data(self):
        assert (
            self.data_base_path.exists()
        ), f"data_base_path does not exist: {self.data_base_path}"

        for idx, lang in enumerate(self.languages):
            wds_base_path = self.data_base_path / "wds" / lang
            assert (
                wds_base_path.exists()
            ), f"WDS path for langauge '{lang}' does not exist: {wds_base_path}"

            self._dataloaders[lang] = self._build_dataloader(lang=lang)
            self.idx2lang[idx] = lang
            log.info(f"Built MaRVL WDS dataloader for language '{lang}'!")

    def _build_dataloader(self, lang: str) -> wds.WebLoader:
        wds = build_wds(
            wds_path=self.data_base_path / "wds" / lang,
            decode="pil",
            tuple_content=(
                "jpg",
                "hypothesis",
                "label_str",
                "caption",
                "lang",
                "sample_id",
                "__key__",
            ),
            batch_size=self.batch_size,
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
            hypotheses = list(map(_decode, batch[1]))
            label_strs = list(map(_decode, batch[2]))
            captions = list(map(_decode, batch[3]))
            languages = list(map(_decode, batch[4]))
            sample_ids = list(map(_decode, batch[5]))
            keys = list(map(_decode, batch[6]))

            languages = [self.languages[lang] for lang in languages]
            prompts = [self.prompt_template.format(HYPOTHESIS=h) for h in hypotheses]

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
                    "hypotheses": hypotheses,
                    "gold_text": label_strs,
                    "prompts": prompts,
                    "captions": captions,
                    "languages": languages,
                    "sample_ids": sample_ids,
                    "keys": keys,
                }
            else:
                pixel_values = self.image_processor(images)  # type: ignore
                prompt_prepro = self.text_processor(prompts)  # type: ignore

                return {
                    "input_ids": prompt_prepro.input_ids,  # B x S
                    "attention_mask": prompt_prepro.attention_mask,  # B x S
                    "pixel_values": pixel_values,  # B x C x H x W
                    "hypotheses": hypotheses,
                    "gold_text": label_strs,
                    "prompts": prompts,
                    "captions": captions,
                    "languages": languages,
                    "sample_ids": sample_ids,
                    "keys": keys,
                }

        return build_wds_dataloader(
            wds,
            num_workers=self.num_workers,
            collate_fn=_collate_fn if not self.no_collate else None,
        )

    def get_lang_id(self, lang_idx: int) -> str:
        return self.idx2lang[lang_idx]

    def get_language_dataloader(self, split: str, lang_id: str) -> wds.WebLoader:
        return self._dataloaders[split][lang_id]

    def train_dataloader(self) -> List[wds.WebLoader]:
        raise ValueError("Train data not configured!")

    def val_dataloader(self) -> List[wds.WebLoader]:
        raise ValueError("Val data not configured!")

    def test_dataloader(self) -> List[wds.WebLoader]:
        return list(self._dataloaders.values())
