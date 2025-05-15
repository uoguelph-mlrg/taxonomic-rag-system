"""
Module of tests for the vision models in the `taxonomic_rag_system` project.

The tests validate the functionality of the following classes:
- `TaxClassifierVLM`: Ensures the taxonomy generation from an image works as expected.
- `DescriptiveCaptioner`: Ensures the caption generation for an image works as expected.

Test cases use mock objects to simulate the behavior of external dependencies.
"""

import pytest

from taxonomic_rag_system.utils.vision_models import (
    DescriptiveCaptioner,
    TaxClassifierVLM,
)


@pytest.mark.asyncio
async def test_tax_classifier_vlm_generate_taxonomy(mock_tax_classifier_pyd):
    """Test taxonomy generation using TaxClassifierVLM."""
    model = TaxClassifierVLM(cap=mock_tax_classifier_pyd)
    result = await model.generate_taxonomy("mock_image_b64")
    assert result["Kingdom"] == "Animalia"
    assert result["Phylum"] == "Chordata"


@pytest.mark.asyncio
async def test_descriptive_captioner_generate_caption(mock_captioner):
    """Test caption generation using DescriptiveCaptioner."""
    model = DescriptiveCaptioner(cap=mock_captioner)
    result = await model.generate_caption("mock_image_b64")
    assert result == "Mock caption"
