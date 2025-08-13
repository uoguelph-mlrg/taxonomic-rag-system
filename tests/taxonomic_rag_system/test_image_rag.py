"""Module of tests for the ImageRAGModel and NaiveVLModel classes."""

import pytest

from taxonomic_rag_system.core.image_rag import ImageRAGModel, NaiveVLModel


@pytest.mark.asyncio
async def test_query_image(mock_captioner, mock_rag_model, mock_image_processor):
    """Test querying an image using ImageRAGModel."""
    from unittest.mock import patch

    with (
        patch(
            "taxonomic_rag_system.core.image_rag.WikiStellaRAGModel",
            return_value=mock_rag_model,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.DescriptiveCaptioner",
            return_value=mock_captioner,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.ImageProcessor",
            return_value=mock_image_processor,
        ),
    ):
        model = ImageRAGModel(
            vstore_path="/mock/path",
            cap=mock_captioner,
            model="mock_model",
        )
        result = await model.query_image(image_path="/mock/image.jpg")
        assert result["caption"] == "Mock caption"
        assert result["results"].classification["Kingdom"] == "Animalia"


@pytest.mark.asyncio
@pytest.mark.integration_test()
async def test_naive_vlm_pipeline(mock_image_processor, mock_tax_classifier):
    """Test the NaiveVLModel pipeline with mock dependencies."""
    from unittest.mock import patch

    with (
        patch(
            "taxonomic_rag_system.core.image_rag.ImageProcessor",
            return_value=mock_image_processor,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.TaxClassifierVLM",
            return_value=mock_tax_classifier,
        ),
    ):
        # Initialize NaiveVLModel with mocks
        naive_vlm = NaiveVLModel(model="mock_model")

        # Run the pipeline
        result = await naive_vlm.query(image_path="/mock/image.jpg")

        # Assertions
        assert result["Kingdom"] == "Animalia"
        assert result["Phylum"] == "Chordata"


@pytest.mark.asyncio
@pytest.mark.integration_test()
async def test_image_rag_pipeline(mock_image_processor, mock_captioner, mock_rag_model):
    """Test the ImageRAGModel pipeline with mock dependencies."""
    from unittest.mock import patch

    with (
        patch(
            "taxonomic_rag_system.core.image_rag.WikiStellaRAGModel",
            return_value=mock_rag_model,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.DescriptiveCaptioner",
            return_value=mock_captioner,
        ),
        patch(
            "taxonomic_rag_system.core.image_rag.ImageProcessor",
            return_value=mock_image_processor,
        ),
    ):
        # Initialize ImageRAGModel with mocks
        image_rag = ImageRAGModel(
            vstore_path="/mock/path",
            cap=mock_captioner,
            model="mock_model",
        )

        # Run the pipeline
        result = await image_rag.query_image(image_path="/mock/image.jpg", context=True)

        # Assertions
        assert result["caption"] == "Mock caption"
        assert result["results"].classification["Kingdom"] == "Animalia"
        assert "Mock context" in result["context"]
