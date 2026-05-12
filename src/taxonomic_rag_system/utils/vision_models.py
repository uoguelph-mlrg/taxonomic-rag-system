"""
Generates captions and taxonomic classifications for images of organisms using VLMs.

Includes base classes for VLMs and specific implementations for generation.
Generating detailed captions or taxonomic classifications from images.

Classes:
--------
- VLM: A base class for vision-language models.
- InstructorVLModel: A base class for VLMs with instructor integration.
- TaxClassifierVLM: A model generating taxonomic classifications of organisms.
- DescriptiveCaptioner: A model generating detailed captions for images of organisms.

Dependencies:
-------------
- instructor: For integrating instructor-based models, used in captioning.
- openai.AsyncOpenAI: For interacting with OpenAI's API.
- taxonomic_rag_system.utils.out_models: For data models like Caption and Tax.

Environment Variables:
----------------------
(set in this script reading from home directory)
- OPENAI_API_KEY: API key for OpenAI.
- OPENROUTER_API_KEY: API key for OpenRouter.

Usage:
------
This module is designed to be used as part of the Taxonomic RAG System for analyzing
and describing organisms in images. It provides both taxonomic classification and
detailed caption generation functionalities depending on the model used.
"""

import json
import logging
import os
from typing import Any, Optional

import instructor
from openai import AsyncOpenAI

from taxonomic_rag_system.utils.helpers import load_api_keys

# Local imports
from taxonomic_rag_system.utils.out_models import Caption, Tax

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = Any  # type: ignore[assignment]


# Configure logging
logging.basicConfig(level=logging.INFO)


class VLM:
    """
    A base class for vision-language models.

    Attributes
    ----------
        cap: The openai model instance.
        model: The model name to be used for generation.
        temp: The temperature setting for generation.
    """

    def __init__(self, cap: Any, model: str, temp: float) -> None:
        self.cap = cap
        self.model = model
        self.temp = temp


class InstructorVLModel(VLM):
    """
    A base class for vision-language models with instructor.

    Adds the instructor 'client' attribute.

    Attributes
    ----------
        cap: The openai model instance.
        client: The instructor client wrapping the cap instance.
        model: The model name to be used for caption generation.
        temp: The temperature setting for generation randomness.
    """

    def __init__(self, cap: Any, model: str, temp: float) -> None:
        super().__init__(cap, model, temp)
        load_api_keys()
        self.client = instructor.from_openai(self.cap)


class TaxClassifierVLM(VLM):
    """
    A VLM model from OpenRouter for taxonomic classification of organisms from images.

    Assumes the image is not too large for the model and contains a single organism.
    Generates a taxonomic classification for that organism.

    Inherits from:
        VLModel

    Attributes
    ----------
        cap: The openai model client configured for OpenRouter.
        model: The model name, default is 'google/gemini-2.0-flash-001'.
        temp: The temperature setting, default is 0.

    Methods
    -------
        generate_caption(image_b64): Generates a taxonomic classification for the image.
    """

    def __init__(self, cap: Any = None, model: str = "", temp: float = 0) -> None:
        load_api_keys()
        if cap is None:
            cap = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                default_headers={
                    "HTTP-Referer": "https://github.com/uoguelph-mlrg/taxonomic-rag-system",
                    "X-Title": "Taxonomic RAG Classifier",
                },
            )
        super().__init__(cap, model, temp)
        self.system_prompt = (
            """
            You are an expert AI vision assistant to a taxonomist. Examine the image, analyze the primary organism's features, and provide a taxonomic classification.

            **Response Format:**
            - Respond **ONLY** with a JSON object containing a `classification` key.
            - The value of `classification` must be a dictionary with keys:
            ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'].
            - Use scientific names or if uncertain, use 'N/A'.

            Example:
            {
            "classification": {
                "Kingdom": "Animalia",
                "Phylum": "Arthropoda",
                "Class": "Insecta",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A"
            }
            }
            """
            + "..." * 256
        )

    async def generate_taxonomy(self, image_b64: str) -> dict[str, str]:
        """
        Generate a taxonomic classification for the primary organism in the image.

        Args:
            image_b64: Base64 encoded image data.

        Returns
        -------
            A dictionary representing the taxonomic classification.
        """
        try:
            return await self._taxonomist(image_b64)
        except Exception as e:
            print(f"Error during caption generation: {e}")
            return {
                "Kingdom": "Animalia",
                "Phylum": "N/A",
                "Class": "N/A",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            }

    async def _taxonomist(self, image_b64: str) -> dict[str, str]:
        """
        Parse base64 image to taxonomic classification.

        The model response is a valid JSON object. It is then parsed to an instance
        of the Tax class, and the classification dictionary is returned.

        Args:
            image_b64: Base64 encoded image data.

        Returns
        -------
            A dictionary containing the taxonomic classification.
        """
        raw_resp = await self.cap.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a taxonomic classification for the primary organism visible in the following image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )

        # Log the raw response for inspection
        logging.debug(f"Raw response: {raw_resp}")

        # Extract the JSON string
        json_content = raw_resp.choices[0].message.content

        # Parse the JSON string to a Python dictionary
        parsed_data = json.loads(json_content)

        # Manually parse the response to a Tax object
        tax = Tax.model_validate(parsed_data)
        assert isinstance(tax, Tax)
        return {key: val for key, val in tax.classification.items() if key != "Domain"}


class DescriptiveCaptioner(InstructorVLModel):
    """
    A model for generating detailed captions for images of organisms.

    Uses OpenAI Instructor instances and assumes the image is appropriately sized.

    Inherits from:
        InstructorVLModel

    Attributes
    ----------
        client: The AI client instance.
        model: The model name, default is 'gpt-4o'.
        temp: The temperature setting, default is 0.
    """

    def __init__(
        self,
        cap: Any,
        model: str,
        temp: float = 0,
    ) -> None:
        if cap is None:
            cap = AsyncOpenAI()
        super().__init__(cap, model, temp)
        self.system_prompt = (
            """
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
        """
            + "..." * 256
        )

    async def generate_caption(self, image_b64: str) -> str:
        """
        Generate a detailed biocaption for the image.

        Args:
            image_b64: Base64 encoded image data.

        Returns
        -------
            A string representing the detailed caption.
        """
        try:
            return await self._caption(image_b64)
        except Exception as e:
            print(f"Error during caption generation: {e}")
            return ""

    async def _caption(self, image_b64: str) -> str:
        """
        Process image with VLM to generate captions.

        Args:
            image_b64: Base64 encoded image data.

        Returns
        -------
            A string caption containing the detailed description of image features.
        """
        raw_resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Write an exhaustive and detailed caption for this image, describing every observable feature thoroughly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            response_model=Caption,
        )
        assert isinstance(raw_resp, Caption)
        return raw_resp.caption


class Qwen3VLDescriptiveCaptionerLocal:
    """Generate detailed captions with a local Qwen3-VL model (HF Transformers).

    This class is intentionally synchronous because HF generation APIs are sync; call
    it via `asyncio.to_thread(...)` from async code paths when needed.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        device_map: str = "cuda0",
        torch_dtype: str = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        prompt: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)

        # Reuse the core intent of the OpenAI caption prompt.
        self.prompt = prompt or (
            "Write an exhaustive and extremely detailed caption for the organism(s) "
            "in this image, describing every observable morphological feature "
            "(body/head shape, appendages, colour pattern, texture, wings, antennae, "
            "eyes, segmentation, etc.) and the surrounding environmental context. "
            "Avoid emotional or non-visible descriptors; rely only on visual cues. "
            "Avoid common names; prefer scientific descriptions. Aim for at least 7 "
            "sentences, with no upper limit if detail demands it."
        )

        self._model: Any | None = None
        self._processor: Any | None = None

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing local VLM dependencies. Install with "
                "`uv sync --extra vlm` (and --extra gpu if applicable)."
            ) from exc

        dtype_obj: Any
        if self.torch_dtype in ("auto", "", None):  # type: ignore[comparison-overlap]
            dtype_obj = "auto"
        else:
            dt = str(self.torch_dtype).lower()
            if dt in ("bf16", "bfloat16"):
                dtype_obj = torch.bfloat16
            elif dt in ("fp16", "float16", "half"):
                dtype_obj = torch.float16
            elif dt in ("fp32", "float32"):
                dtype_obj = torch.float32
            else:
                raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype!r}")

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=dtype_obj,
            attn_implementation=self.attn_implementation,
            trust_remote_code=bool(self.trust_remote_code),
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=bool(self.trust_remote_code)
        )

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption from a PIL image."""
        self._lazy_load()
        assert self._model is not None
        assert self._processor is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Some processors include token_type_ids which the model may not accept.
        try:
            inputs.pop("token_type_ids", None)
        except Exception:
            pass

        # Best-effort device placement for the common single-GPU case.
        try:
            import torch

            model_device = getattr(self._model, "device", None)
            if model_device is not None and str(model_device) != "meta":
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(model_device)
        except Exception:
            # If device_map is complex, rely on HF internals.
            pass

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
        }
        # Only enable sampling knobs when do_sample is requested.
        if self.do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )

        generated_ids = self._model.generate(**inputs, **gen_kwargs)
        # Trim the prompt tokens.
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated_ids = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
        out = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # batch_decode may return list[str] or list[list[str]] depending on tokenizer.
        if not out:
            return ""
        first = out[0]
        if isinstance(first, list):
            return first[0] if first else ""
        return str(first)
