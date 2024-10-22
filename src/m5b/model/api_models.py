import base64
import io
import os
from typing import List, Literal

import backoff
import google.ai.generativelanguage as glm
import google.generativeai as genai
import vertexai
from img2dataset.resizer import ResizeMode, Resizer
from openai import BadRequestError, OpenAI
from PIL import Image
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, Part


class ApiModelBase:
    def __init__(
        self,
        model_id: str,
        max_img_size: int = 640,
    ):
        self.model_id = model_id
        self._resizer = Resizer(
            image_size=max_img_size,
            resize_mode=ResizeMode.keep_ratio,
            resize_only_if_bigger=True,
        )

    def _resize_image_bytes(self, image_bytes: io.BytesIO) -> bytes:
        resized_img_stream, _, _, _, _, error = self._resizer(image_bytes)
        if error:
            raise ValueError(f"Error resizing image: {error}")
        return io.BytesIO(resized_img_stream).getvalue()

    def image_to_bytes(self, image: Image.Image) -> bytes:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        return self._resize_image_bytes(image_bytes)

    def _create_image_parts(
        self, images: Image.Image | List[Image.Image]
    ) -> List[glm.Part | Part]:
        raise NotImplementedError

    def generate_content(
        self,
        prompt: str,
        images: Image.Image | List[Image.Image],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        raise NotImplementedError


class GeminiProVisionVertexAIModel(ApiModelBase):
    def __init__(
        self,
        project_id: str | None,
        model_id: str = "gemini-1.0-pro-vision-001",
        max_img_size: int = 640,
    ):
        super().__init__(
            model_id,
            max_img_size,
        )
        if project_id is None:
            project_id = os.environ.get("GOOGLE_PROJECT_ID", None)
        if project_id is None:
            raise ValueError("Please provide a Google project ID!")
        vertexai.init(project=project_id)
        self.model = GenerativeModel("gemini-1.0-pro-vision-001")

    def _create_image_parts(
        self, images: Image.Image | List[Image.Image]
    ) -> List[Part]:
        if not isinstance(images, list):
            images = [images]
        return [
            Part.from_data(
                data=self.image_to_bytes(image),
                mime_type="image/jpeg",
            )
            for image in images
        ]

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def generate_content(
        self,
        prompt: str,
        images: Image.Image | List[Image.Image],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        gen_config = GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_new_tokens,
            # FIXME: This sometimes causes an error
            # temperature=temperature,
            # top_k=top_k,
            # top_p=top_p,
        )
        response = self.model.generate_content(
            [
                Part.from_text(prompt),
                *self._create_image_parts(images),
            ],
            generation_config=gen_config,
        )

        try:
            return response.text.strip()
        except Exception as e:
            if isinstance(e, IndexError) or (
                isinstance(e, ValueError)
                and (
                    "Content has no parts." in str(e)
                    or "Multiple content parts are not supported." in str(e)
                )
            ):
                print(f"Error when getting response content: {e}")
            return ""


class GeminiProVisionGoogleStudioModel(ApiModelBase):
    def __init__(
        self,
        api_key: str | None,
        model_id: str = "gemini-pro-vision",
        max_img_size: int = 640,
    ):
        super().__init__(
            model_id,
            max_img_size,
        )

        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY", None)
        if api_key is None:
            raise ValueError("Please provide a Google API key!")
        genai.configure(api_key=api_key)

        available_models = {
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        }

        if "models/" + model_id not in available_models:
            raise ValueError(
                f"Model 'models/{model_id}' is not available! Available models: {available_models}"
            )

        self.model = genai.GenerativeModel(self.model_id)

    def _create_image_parts(
        self, images: Image.Image | List[Image.Image]
    ) -> List[glm.Part]:
        if not isinstance(images, list):
            images = [images]
        return [
            glm.Part(
                inline_data=glm.Blob(
                    mime_type="image/jpeg",
                    data=self.image_to_bytes(image),
                )
            )
            for image in images
        ]

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def generate_content(
        self,
        prompt: str,
        images: Image.Image | List[Image.Image],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        gen_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        response = self.model.generate_content(
            glm.Content(
                parts=[
                    glm.Part(
                        text=prompt,
                    ),
                    *self._create_image_parts(images),
                ],
            ),
            generation_config=gen_config,
        )

        try:
            return response.parts[0].text.strip()
        except Exception as e:
            if (isinstance(e, IndexError) or isinstance(e, ValueError)) and (
                "The `response.parts` quick accessor" in str(e)
                or "Multiple content parts are not supported." in str(e)
            ):
                print(f"Error when getting response content: {e}")
            return ""


class GPT4VisionOpenAIModel(ApiModelBase):
    def __init__(
        self,
        api_key: str | None,
        model_id: str = "gpt-4-vision-preview",
        max_img_size: int = 640,
        img_detail: Literal["low", "auto", "high"] = "low",
    ):
        super().__init__(
            model_id,
            max_img_size,
        )

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("Please provide an OpenAI API key!")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.img_detail = img_detail

    def _create_image_parts(self, images: Image.Image | List[Image.Image]) -> List:
        if not isinstance(images, list):
            images = [images]
        base64_images = [
            base64.b64encode(self.image_to_bytes(image)).decode("utf-8")
            for image in images
        ]
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": self.img_detail,
                },
            }
            for base64_image in base64_images
        ]

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def generate_content(
        self,
        prompt: str,
        images: Image.Image | List[Image.Image],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *self._create_image_parts(images),
                        ],
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            if isinstance(e, BadRequestError) and e.code == "content_policy_violation":
                print(f"Content Policy Violation!: {e}")
                return ""
            raise e

        try:
            return response.choices[0].message.content.strip()
        except Exception as e:
            if isinstance(e, (ValueError, IndexError)):
                print(f"Error when getting response content: {e}")
            return ""


def load_api_model(
    model_id: str,
    api_model_key: str | None,
    google_project_id: str | None,
    gpt_4_img_detail: Literal["low", "auto", "high"] = "low",
) -> (
    GeminiProVisionGoogleStudioModel
    | GeminiProVisionVertexAIModel
    | GPT4VisionOpenAIModel
):
    if model_id in [
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-1106-vision-preview",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-4o",
    ]:
        model = GPT4VisionOpenAIModel(
            api_key=api_model_key, img_detail=gpt_4_img_detail, model_id=model_id
        )
    elif model_id == "gemini-pro-vision":
        model = GeminiProVisionVertexAIModel(project_id=google_project_id)
    elif model_id == "gemini-pro-vision-google-studio":
        model = GeminiProVisionGoogleStudioModel(api_key=api_model_key)
    else:
        raise ValueError(f"Model ID {model_id} not supported!")

    print(f"Loaded model {model.__class__.__name__}: {model_id}")
    return model
