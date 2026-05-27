"""
Module of tests for the vision models in the `taxonomic_rag_system` project.

The tests validate the functionality of the following classes:
- `TaxClassifierVLM`: Ensures the taxonomy generation from an image works as expected.
- `DescriptiveCaptioner`: Ensures the caption generation for an image works as expected.

Test cases use mock objects to simulate the behavior of external dependencies.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taxonomic_rag_system.utils.out_models import Caption
from taxonomic_rag_system.utils.vision_models import (
    DescriptiveCaptioner,
    TaxClassifierVLM,
)


@pytest.mark.asyncio
async def test_tax_classifier_vlm_generate_taxonomy(tmp_path):
    """Test taxonomy generation using TaxClassifierVLM."""
    # Mock the response structure for JSON format
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(
        {
            "classification": {
                "Kingdom": "Animalia",
                "Phylum": "Chordata",
                "Class": "Mammalia",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            }
        }
    )
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_logprob_token = MagicMock()
    mock_logprob_token.token = "{"
    mock_logprob_token.logprob = -0.1
    mock_logprob_token.top_logprobs = []
    mock_logprobs = MagicMock()
    mock_logprobs.content = [mock_logprob_token]
    mock_choice.logprobs = mock_logprobs
    mock_response.choices = [mock_choice]
    mock_response.model = "test-model"
    mock_response.created = 123
    mock_response.system_fingerprint = "fp"

    mock_cap = AsyncMock()
    mock_cap.chat.completions.create.return_value = mock_response

    logprobs_path = tmp_path / "logprobs.jsonl"
    model = TaxClassifierVLM(
        cap=mock_cap,
        model="test-model",
        logprobs_path=str(logprobs_path),
    )
    result = await model.generate_taxonomy("mock_image_b64", rsid="RS1")

    assert result["Kingdom"] == "Animalia"
    assert result["Phylum"] == "Chordata"
    assert result["Class"] == "Mammalia"
    assert result["Order"] == "N/A"
    assert logprobs_path.exists()
    record = json.loads(logprobs_path.read_text(encoding="utf-8").strip())
    assert record["rsid"] == "RS1"
    assert record["tokens"]
    create_kwargs = mock_cap.chat.completions.create.await_args.kwargs
    assert create_kwargs["logprobs"] is True
    assert create_kwargs["top_logprobs"] == 20
    # Verify Domain is filtered out if present
    assert "Domain" not in result


@pytest.mark.asyncio
async def test_tax_classifier_vlm_error_handling():
    """Test error handling in taxonomy generation."""
    mock_cap = AsyncMock()
    mock_cap.chat.completions.create.side_effect = Exception("API Error")

    model = TaxClassifierVLM(cap=mock_cap, model="test-model")
    result = await model.generate_taxonomy("mock_image_b64")

    # Should return default values on error
    assert result["Kingdom"] == "Animalia"
    assert result["Phylum"] == "N/A"


@pytest.mark.asyncio
async def test_descriptive_captioner_generate_caption():
    """Test caption generation using DescriptiveCaptioner."""
    # Mock the instructor client response
    mock_caption_obj = Caption(caption="Detailed mock caption of the organism")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_caption_obj)

    mock_cap = AsyncMock()

    with patch("instructor.from_openai", return_value=mock_client):
        model = DescriptiveCaptioner(cap=mock_cap, model="gpt-4o")
        result = await model.generate_caption("mock_image_b64")

    assert result == "Detailed mock caption of the organism"


@pytest.mark.asyncio
async def test_descriptive_captioner_error_handling():
    """Test error handling in caption generation."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    mock_cap = AsyncMock()

    with patch("instructor.from_openai", return_value=mock_client):
        model = DescriptiveCaptioner(cap=mock_cap, model="gpt-4o")
        result = await model.generate_caption("mock_image_b64")

    # Should return empty string on error
    assert result == ""
