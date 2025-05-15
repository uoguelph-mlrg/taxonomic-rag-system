"""Module of tests for the ImageRAGModel and NaiveVLModel classes."""

import pytest

from taxonomic_rag_system.core.image_rag import ImageRAGModel, NaiveVLModel


@pytest.mark.asyncio
async def test_query_image(mock_captioner, mock_rag_model):
    """Test querying an image using ImageRAGModel."""
    model = ImageRAGModel(
        vstore_path="/mock/path",
        cap=mock_captioner,
        model="mock_model",
    )
    model.rag_model = mock_rag_model
    result = await model.query_image(image_path="/mock/image.jpg")
    assert result["caption"] == "Mock caption"
    assert result["results"]["classification"]["Kingdom"] == "Animalia"


@pytest.mark.asyncio
@pytest.mark.integration_test()
async def test_naive_vlm_pipeline(mock_image_processor, mock_tax_classifier):
    """Test the NaiveVLModel pipeline with mock dependencies."""
    # Initialize NaiveVLModel with mocks
    naive_vlm = NaiveVLModel(model="mock_model")
    naive_vlm.image_processor = mock_image_processor
    naive_vlm.model = mock_tax_classifier

    # Run the pipeline
    result = await naive_vlm.query(image_path="/mock/image.jpg")

    # Assertions
    assert result["Kingdom"] == "Animalia"
    assert result["Phylum"] == "Chordata"


@pytest.mark.asyncio
@pytest.mark.integration_test()
async def test_image_rag_pipeline(mock_image_processor, mock_captioner, mock_rag_model):
    """Test the ImageRAGModel pipeline with mock dependencies."""
    # Initialize ImageRAGModel with mocks
    image_rag = ImageRAGModel(
        vstore_path="/mock/path",
        cap=mock_captioner,
        model="mock_model",
    )
    image_rag.image_processor = mock_image_processor
    image_rag.captioner = mock_captioner
    image_rag.rag_model = mock_rag_model

    # Run the pipeline
    result = await image_rag.query_image(image_path="/mock/image.jpg", context=True)

    # Assertions
    assert result["caption"] == "Mock caption"
    assert result["results"]["classification"]["Kingdom"] == "Animalia"
    assert result["context"] == ["Mock context"]
