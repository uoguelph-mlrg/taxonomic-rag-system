"""Tests for the local VLM captioner and local end-to-end injection path."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from taxonomic_rag_system.utils.local_vlm import LocalVLMCaptioner, LocalVLMConfig


def _sample_image_b64() -> str:
    """Create a tiny in-memory RGB image and return it as base64."""
    image = Image.new("RGB", (2, 2), color="green")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _taxonomy_payload() -> str:
    """Build a minimal valid TaxBiodiversity JSON string."""
    return """
    {
      "classification": {
        "Kingdom": "Animalia",
        "Phylum": "Arthropoda",
        "Class": "Insecta",
        "Order": "Coleoptera",
        "Family": "Carabidae",
        "Genus": "Carabus",
        "Species": "Carabus nemoralis"
      },
      "ancestral": "shared morphology",
      "specific": "elytra and body shape",
      "commentary": "best local guess",
      "bio_knowledge": "ground beetles are terrestrial predators"
    }
    """


def test_local_vlm_config_defaults():
    """Local VLM config should preserve the expected defaults."""
    cfg = LocalVLMConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct")

    assert cfg.max_new_tokens == 1024
    assert cfg.device_map == "auto"
    assert cfg.torch_dtype == "auto"
    assert cfg.trust_remote_code is True


def test_local_vlm_captioner_returns_caption_from_mocked_hf_stack():
    """The local VLM captioner should decode model output into a stripped string."""
    captioner = LocalVLMCaptioner(
        LocalVLMConfig(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            device_map="cuda0",
        )
    )
    fake_processor = MagicMock()
    fake_model = MagicMock()
    fake_inputs = MagicMock()
    fake_inputs.input_ids = SimpleNamespace(shape=(1, 4))
    fake_inputs.to.return_value = fake_inputs
    fake_processor.apply_chat_template.return_value = "<prompt>"
    fake_processor.return_value = fake_inputs
    fake_processor.batch_decode.return_value = [" detailed caption "]
    fake_generated_ids = MagicMock()
    fake_generated_ids.__getitem__.return_value = [[1, 2, 3]]
    fake_model.generate.return_value = fake_generated_ids

    with patch.object(captioner, "_ensure_loaded", return_value=(fake_processor, fake_model)):
        caption = captioner._caption_sync(_sample_image_b64())

    assert caption == "detailed caption"
    fake_model.generate.assert_called_once()
    fake_processor.batch_decode.assert_called_once()


@pytest.mark.asyncio
async def test_local_vlm_captioner_returns_empty_string_on_failure():
    """Caption generation errors should degrade to an empty string."""
    captioner = LocalVLMCaptioner(LocalVLMConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct"))

    with patch.object(captioner, "_caption_sync", side_effect=RuntimeError("bad decode")):
        caption = await captioner.generate_caption("broken")

    assert caption == ""


def test_wiki_stella_rag_model_uses_injected_local_llm():
    """The Stella retriever should forward an injected local LLM into the chain."""
    from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel

    local_llm = MagicMock()
    fake_vectorstore = MagicMock()
    fake_vectorstore.as_retriever.return_value = MagicMock()

    with (
        patch.object(
            WikiStellaRAGModel,
            "_set_up_retriever",
            return_value=fake_vectorstore,
        ),
        patch("taxonomic_rag_system.utils.retriever.RAGChainBuilder") as builder_cls,
    ):
        WikiStellaRAGModel(
            vstore_path="/mock/vstore",
            llm=local_llm,
            multiquery=False,
            rerank=False,
        )

    assert builder_cls.call_args.kwargs["llm"] is local_llm


def test_local_only_retriever_path_does_not_require_openai_key():
    """The retriever should skip API-key loading for local-only configurations."""
    from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel

    fake_vectorstore = MagicMock()
    fake_vectorstore.as_retriever.return_value = MagicMock()

    with (
        patch(
            "taxonomic_rag_system.utils.retriever.load_api_keys",
            side_effect=AssertionError("should not load keys"),
        ),
        patch.object(
            WikiStellaRAGModel,
            "_set_up_retriever",
            return_value=fake_vectorstore,
        ),
    ):
        WikiStellaRAGModel(
            vstore_path="/mock/vstore",
            llm=MagicMock(),
            multiquery=False,
            rerank=False,
        )


@pytest.mark.asyncio
async def test_image_rag_uses_injected_captioner(mock_rag_model, mock_image_processor):
    """ImageRAGModel should bypass the OpenAI captioner when a captioner is injected."""
    from taxonomic_rag_system.core.image_rag import ImageRAGModel

    mock_captioner = AsyncMock()
    mock_captioner.generate_caption.return_value = "Injected caption"

    with (
        patch(
            "taxonomic_rag_system.core.image_rag.WikiStellaRAGModel",
            return_value=mock_rag_model,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.ImageProcessor",
            return_value=mock_image_processor,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.DescriptiveCaptioner",
            side_effect=AssertionError("default captioner should not be built"),
        ),
    ):
        model = ImageRAGModel(
            vstore_path="/mock/path",
            captioner=mock_captioner,
            rag_llm=MagicMock(),
        )
        result = await model.query_image(image_path="/mock/image.jpg")

    assert result["caption"] == "Injected caption"
    assert result["results"].classification["Kingdom"] == "Animalia"


def test_rag_chain_builder_retries_with_json_only_suffix():
    """Local LLM parsing should retry once with a stricter JSON-only suffix."""
    from taxonomic_rag_system.utils.retriever import (
        JSON_ONLY_RETRY_SUFFIX,
        RAGChainBuilder,
    )

    class _FakeLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.outputs = ["not json", f"```json\n{_taxonomy_payload()}\n```"]

        def invoke(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return self.outputs.pop(0)

    fake_llm = _FakeLLM()
    builder = RAGChainBuilder(retriever=MagicMock(), llm=fake_llm)

    result = builder.invoke({"context": "ctx", "caption": "cap"})

    assert result.classification["Species"] == "Carabus nemoralis"
    assert fake_llm.prompts[1].endswith(JSON_ONLY_RETRY_SUFFIX)


def test_resolve_llm_choice_applies_gemma_defaults():
    """Gemma model selection should fill in the branch's default generation cap."""
    from taxonomic_rag_system.runs.rare_species_simp_rag_local_vlm_local_llm import (
        _resolve_llm_choice,
    )

    args = SimpleNamespace(
        model_choice="gemma",
        llm_model_id="",
        llm_trust_remote_code=False,
        llm_max_new_tokens=None,
    )

    model_id, trust_remote_code, max_new_tokens = _resolve_llm_choice(args)

    assert model_id == "google/gemma-7b-it"
    assert trust_remote_code is False
    assert max_new_tokens == 512
