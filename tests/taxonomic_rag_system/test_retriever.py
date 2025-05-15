"""Module of tests for retriever classes in `taxonomic_rag_system` project."""

import pytest

from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel


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
