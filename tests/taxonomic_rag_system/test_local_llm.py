"""Tests for local Hugging Face generation config handling."""

import json

import pytest

from taxonomic_rag_system.runs.rare_species_simp_rag_from_prompts_local_llm import (
    _invoke_structured_taxonomy_response,
)
from taxonomic_rag_system.utils.local_llm import (
    LocalLLMConfig,
    _apply_generation_defaults,
    normalize_device_map,
)


def test_apply_generation_defaults_uses_model_defaults_when_values_are_none():
    """Unset generation fields should not override the model defaults."""
    cfg = LocalLLMConfig(model_id="dummy-model")

    kwargs = _apply_generation_defaults({}, cfg)

    assert kwargs == {"return_full_text": False}


def test_apply_generation_defaults_omits_sampling_args_for_greedy_decoding():
    """Greedy decoding should not pass invalid sampling-only arguments."""
    cfg = LocalLLMConfig(model_id="dummy-model", temperature=0.0, top_p=1.0)

    kwargs = _apply_generation_defaults({}, cfg)

    assert kwargs["do_sample"] is False
    assert kwargs["return_full_text"] is cfg.return_full_text
    assert "max_new_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_normalize_device_map_single_gpu_aliases():
    """Single-GPU aliases should map to a single-device dict for HF."""
    assert normalize_device_map("cuda0") == {"": 0}
    assert normalize_device_map("GPU0") == {"": 0}
    assert normalize_device_map("single") == {"": 0}
    assert normalize_device_map("0") == {"": 0}
    assert normalize_device_map("cuda:0") == {"": 0}
    assert normalize_device_map({"": 0}) == {"": 0}


def test_normalize_device_map_auto_and_json():
    """Auto and JSON strings should pass through or parse."""
    assert normalize_device_map("auto") == "auto"
    assert normalize_device_map('{"": 0}') == {"": 0}


def test_normalize_device_map_invalid_json_raises():
    """Malformed JSON starting with { should raise ValueError."""
    with pytest.raises(ValueError, match="device_map"):
        normalize_device_map("{not json")


def test_apply_generation_defaults_keeps_sampling_args_when_enabled():
    """Sampling runs should still receive temperature and top-p defaults."""
    cfg = LocalLLMConfig(
        model_id="dummy-model",
        max_new_tokens=900,
        temperature=0.7,
        top_p=0.9,
    )

    kwargs = _apply_generation_defaults({}, cfg)

    assert kwargs["do_sample"] is True
    assert kwargs["max_new_tokens"] == 900
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


class _FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def test_invoke_structured_taxonomy_response_retries_with_json_only_suffix():
    """Second pass should succeed when the retry prompt fixes formatting."""
    llm = _FakeLLM(
        [
            "not json at all",
            json.dumps(
                {
                    "classification": {
                        "Kingdom": "Animalia",
                        "Phylum": "Arthropoda",
                        "Class": "Insecta",
                        "Order": "Odonata",
                        "Family": "Gomphidae",
                        "Genus": "Hemigomphus",
                        "Species": "Hemigomphus theischingeri",
                    },
                    "ancestral": "shared traits",
                    "specific": "specific traits",
                    "commentary": "reasoning",
                    "bio_knowledge": "knowledge",
                }
            ),
        ]
    )

    response = _invoke_structured_taxonomy_response(llm, "classify this organism")

    assert response["classification"]["Species"] == "Hemigomphus theischingeri"
    assert llm.prompts[0] == "classify this organism"
    assert llm.prompts[1].endswith(
        "\n\nOutput ONLY the JSON object. Do not use markdown.\n"
    )


def test_invoke_structured_taxonomy_response_raises_instead_of_fallback_output():
    """Inference failures should stop the run rather than emit placeholder labels."""
    llm = _FakeLLM(
        [
            RuntimeError("cuda/cpu mismatch"),
            RuntimeError("cuda/cpu mismatch"),
        ]
    )

    with pytest.raises(RuntimeError, match="refusing to emit fallback taxonomy"):
        _invoke_structured_taxonomy_response(llm, "classify this organism")
