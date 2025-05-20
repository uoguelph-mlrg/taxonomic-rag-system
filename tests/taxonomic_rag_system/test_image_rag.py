"""Module of tests for the ImageRAGModel and NaiveVLModel classes."""

from unittest.mock import mock_open, patch

import pytest

from taxonomic_rag_system.core.image_rag import ImageRAGModel, NaiveVLModel


@pytest.fixture(autouse=True)
def mock_api_key_files():
    """
    Mock the behavior of opening a file to read an API key.

    This function uses `unittest.mock.patch` to replace the built-in `open` function
    with a mock that returns a predefined string ("mock_api_key") when read. It is
    useful for testing code that relies on reading API keys from files without
    requiring actual files to be present.

    Yields
    ------
        None: This is a generator function that provides a mocked context for the
        duration of its usage.
    """
    with patch("builtins.open", mock_open(read_data="mock_api_key")):
        yield


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
