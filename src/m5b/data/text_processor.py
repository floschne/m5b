import re
from typing import Any, Dict, List

from numpy import ndarray
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding

from m5b.util.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TextProcessor:
    def __init__(self, hf_tokenizer_id: str, **extra_kwargs):
        log.info(f"Loading TextProcessor {hf_tokenizer_id}")
        self.hf_tokenizer_id = hf_tokenizer_id

        tok_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": hf_tokenizer_id,
        }
        if hf_tokenizer_id.startswith("DAMO-NLP-MT/polylm"):
            tok_kwargs.update(
                **{
                    "legacy": False,
                    "use_fast": False,
                }
            )

        self.tokenizer = AutoTokenizer.from_pretrained(**tok_kwargs, **extra_kwargs)

        if self.tokenizer._pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self,
        text: str | List[str],
        padding: str = "longest",
        return_tensors: str = "pt",
        check_prompt: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if check_prompt and self.hf_tokenizer_id.startswith("llava-hf"):
            self._check_llava_prompt(text)
        if self.hf_tokenizer_id.startswith("mistralai/Mistral-7B-Instruct"):
            if check_prompt:
                self._check_mistral_instruct_prompt(text)
            # we add the special tokens manually in  the prompt template
            kwargs.update({"add_special_tokens": False})
        return self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, **kwargs
        )

    def _check_llava_prompt(self, text: str | List[str]) -> None:
        if isinstance(text, list):
            for t in text:
                self._check_llava_prompt(t)
            return
        prompt_match = re.match(r"USER: <image>\n.*\nASSISTANT:", text)
        if prompt_match is None:
            log.warning("The prompt pattern is not matched!")

    def _check_mistral_instruct_prompt(self, text: str | List[str]) -> None:
        if isinstance(text, list):
            for t in text:
                self._check_mistral_instruct_prompt(t)
            return
        prompt_match = re.match(r"\<s\>\[INST\] .* \[\/INST\]", text)
        if prompt_match is None:
            log.warning("The prompt pattern is not matched!")

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id  # type: ignore

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id  # type: ignore

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def batch_decode(
        self,
        sequences: List[int] | List[List[int]] | ndarray | Tensor | Any,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,  # type: ignore
            **kwargs,
        )
