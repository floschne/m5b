from random import choice
from typing import List


class CaptionPrompter:
    def __init__(
        self,
        prompt_verbs: List[str] = [
            "Caption",
            "Describe",
            "Explain",
            "Summarize",
            "Tell me about",
        ],
        prompt_objects: List[str] = [
            "the image",
            "the picture",
            "the photo",
            "the scene",
            "the photograph",
        ],
        prompt_adverbs: List[str] = [
            "briefly",
            "shortly",
            "in a few words",
            "in a sentence",
            "",
        ],
        # default args
        random: bool = True,
        prompt_num: int = 0,
        colon: bool = False,
        prompt_template: str | None = None,
    ):
        self.prompt_verbs = prompt_verbs
        self.prompt_objects = prompt_objects
        self.prompt_adverbs = prompt_adverbs
        self.prompt_template = prompt_template

        if prompt_template is not None and "{PROMPT}" not in prompt_template:
            raise ValueError(
                "Prompt template must include the string '{PROMPT}' to be replaced"
            )

        self.prompts = []
        for verb in self.prompt_verbs:
            for obj in self.prompt_objects:
                for adv in self.prompt_adverbs:
                    self.prompts.append(f"{verb} {obj} {adv}")

        self.num_prompts = len(self.prompts)

        self.random = random
        self.prompt_num = prompt_num
        self.colon = colon

        # TODO add translation support

    def _add_lang_and_colon(self, prompt: str, lang: str | None, colon: bool) -> str:
        if lang is not None:
            prompt += f" in {lang}"
        if colon:
            prompt += ":"
        return prompt

    def generate_prompt(
        self,
        random: bool | None = None,
        prompt_num: int | None = None,
        lang: str | None = None,
        colon: bool | None = None,
    ) -> str:
        if random is None:
            random = self.random
        if prompt_num is None:
            prompt_num = self.prompt_num
        if colon is None:
            colon = self.colon
        if random:
            prompt = self._add_lang_and_colon(choice(self.prompts), lang, colon)
        else:
            prompt = self._add_lang_and_colon(self.prompts[prompt_num], lang, colon)

        if self.prompt_template is not None:
            prompt = self.prompt_template.format(PROMPT=prompt)

        return prompt

    def generate_prompts(
        self,
        langs: List[str],
        random: bool | None = None,
        prompt_num: int | None = None,
        colon: bool | None = None,
    ) -> List[str]:
        return [
            self.generate_prompt(
                random=random,
                prompt_num=prompt_num,
                lang=lang,
                colon=colon,
            )
            for lang in langs
        ]
