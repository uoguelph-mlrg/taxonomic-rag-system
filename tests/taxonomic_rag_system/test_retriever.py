"""Module of tests for retriever classes in `taxonomic_rag_system` project."""

from unittest.mock import mock_open, patch

import pytest

from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel


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


def test_wiki_stella_rag_model_initialization():
    """Test initialization of WikiStellaRAGModel."""
    model = WikiStellaRAGModel(
        vstore_path="/mock/path",
        collection_name="mock_collection",
        embedding_model="mock_model",
        search_type="similarity",
        k=10,
    )
    assert model.collection_name == "mock_collection"
    assert model.k == 10


@pytest.mark.asyncio
@pytest.mark.integration_test()
async def test_retriever_pipeline(mock_rag_model):
    """Test the retriever pipeline with a mock RAG model."""
    retriever = WikiStellaRAGModel(
        vstore_path="/mock/path",
        collection_name="mock_collection",
        embedding_model="mock_model",
    )
    retriever.model = mock_rag_model

    result = await retriever.ainvoke("Mock caption")
    assert result["classification"]["Kingdom"] == "Animalia"
