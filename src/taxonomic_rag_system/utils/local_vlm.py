"""Local Hugging Face VLM utilities for descriptive image captioning."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from taxonomic_rag_system.utils.local_llm import normalize_device_map


DESCRIPTIVE_CAPTION_SYSTEM_PROMPT = """
You are an expert AI vision assistant to a taxonomist that describes animals in images.

Your task is to describe in extensive detail all the physical features (body and head shape, appendages, colour pattern, shape, texture, etc) of any organism(s) observable in the image.

Ensure each feature is elaborated upon wherever possible. Elaborate on the visual traits and morphology of subsections of the organism such as any appendages (such as limbs, wings etc) that are visible in the image — aim to describe them thoroughly.

Additionally, include a comprehensive description of the current state and/or life stage of the organism, with nuances in coloration, wear, or any distinctive features.

Please describe the environmental context surrounding the animal in detail. For example, if a butterfly is resting on flowers, you should also delve into the unique characteristics of the flowers, such as their shape, color, and arrangement.

Avoid using species common names (such as 'Monarch' for a butterfly).

Aim for a detailed analysis of at least 7 sentences with no upper limit if the detail demands it.

Do not include emotional descriptors (such as 'peaceful setting') or any other non-visible descriptors. Everything you mention should be evident in the image to an observer. Your response must rely solely on visual cues from the image, avoiding any inferences that are not evident.

Write a extremely detailed caption for the organism in the image and without commenting on the contrast or 'feeling' of the image.

An example of the type of caption you should produce is:
    Insecta with 4 visible jointed legs, partially translucent wings and compound eyes. There is a three-part body with a head, thorax and abdomen. An anterior lateral view of an adult fly with an abdomen that is mostly black and has a black tail-like taper. The wings have streaks of white as does the thorax and are black elsewhere. The prescutum and scutum are brown and in addition to the head, have small shiny hairs. The wings attach at the middle of the thorax, as do the legs. The legs have an initial black segment but are mostly coppery-brown and terminate into a triangular base. The wings are not as long as the length of the body and lay relatively flat at an angle away from the body with 2 segmented translucent halteres. The head is copper, orange and brown with white bordering. The head is visibly segmented from the thorax but the thorax and abdomen appear continuous and not visibly segmented. One brownish-orange eye with a white border is fully visible and the other eye is partially visible. There are two coppery kidney-shaped mouth parts protruding from the lower front of the head. A single shiny antennae is visible. The fly is standing on a green leaf that has pointed edges.
""" + "..." * 256

DESCRIPTIVE_CAPTION_USER_PROMPT = (
    "Write an exhaustive and detailed caption for this image, describing every "
    "observable feature thoroughly."
)


@dataclass(frozen=True)
class LocalVLMConfig:
    """Configuration for a local Hugging Face vision-language captioner."""

    model_id: str
    max_new_tokens: int = 1024
    device_map: str | dict[str, Any] = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True


def _decode_image_b64(image_b64: str) -> Image.Image:
    """Decode a base64 image string into a RGB PIL image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


class LocalVLMCaptioner:
    """Generate descriptive captions using a local Hugging Face Qwen-VL model."""

    def __init__(self, cfg: LocalVLMConfig) -> None:
        self.cfg = cfg
        self._processor: Any | None = None
        self._model: Any | None = None
        self._resolved_device_map: str | dict[str, Any] | None = None

    @staticmethod
    def _resolve_model_class() -> type[Any]:
        """Resolve the best available HF model class for Qwen VL inference."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration

                return Qwen2VLForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForVision2Seq

                return AutoModelForVision2Seq

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Lazily load the processor and model."""
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        from transformers import AutoProcessor

        model_cls = self._resolve_model_class()
        self._resolved_device_map = normalize_device_map(self.cfg.device_map)
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        self._model = model_cls.from_pretrained(
            self.cfg.model_id,
            device_map=self._resolved_device_map,
            torch_dtype=self.cfg.torch_dtype,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        return self._processor, self._model

    def _build_inputs(self, image: Image.Image) -> tuple[Any, Any]:
        """Construct multimodal processor inputs for Qwen-VL caption generation."""
        processor, model = self._ensure_loaded()
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": DESCRIPTIVE_CAPTION_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DESCRIPTIVE_CAPTION_USER_PROMPT},
                ],
            },
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"{DESCRIPTIVE_CAPTION_SYSTEM_PROMPT}\n\n"
                f"{DESCRIPTIVE_CAPTION_USER_PROMPT}"
            )

        model_inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        if self._resolved_device_map == {"": 0} and hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to("cuda:0")
        return model, model_inputs

    def _caption_sync(self, image_b64: str) -> str:
        """Run synchronous local VLM inference and return the decoded caption."""
        processor, _ = self._ensure_loaded()
        image = _decode_image_b64(image_b64)
        model, model_inputs = self._build_inputs(image)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=self.cfg.max_new_tokens,
        )
        input_ids = getattr(model_inputs, "input_ids", None)
        if input_ids is not None:
            generated_ids = generated_ids[:, input_ids.shape[-1] :]
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return caption.strip()

    async def generate_caption(self, image_b64: str) -> str:
        """Generate a detailed caption asynchronously."""
        try:
            return await asyncio.to_thread(self._caption_sync, image_b64)
        except Exception as e:
            print(f"Error during caption generation: {e}")
            return ""
