"""
Unit and integration tests for the ImageRAGModel and NaiveVLModel classes.

This module tests the core image processing and RAG pipeline functionality
in the taxonomic_rag_system project. It includes both unit tests with mocked
dependencies and integration tests that validate the complete pipeline flow.

Tested classes:
- `ImageRAGModel`: Complete RAG pipeline combining image captioning and retrieval
- `NaiveVLModel`: Simple vision-language model for direct taxonomic classification

Test types:
- Unit tests: Individual method testing with mocked external dependencies
- Integration tests: End-to-end pipeline testing with mock services

The tests use fixtures from conftest.py to mock external API calls and ensure
reproducible test results without external service dependencies.
"""

from unittest.mock import patch

import pytest

from taxonomic_rag_system.core.image_rag import ImageRAGModel, NaiveVLModel


@pytest.mark.asyncio
async def test_query_image(mock_captioner, mock_rag_model, mock_image_processor):
    """Test querying an image using ImageRAGModel."""
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
