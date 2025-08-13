"""Module of tests for retriever classes in `taxonomic_rag_system` project."""

import pytest

from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel


def test_wiki_stella_rag_model_initialization():
    """Test initialization of WikiStellaRAGModel."""
    from unittest.mock import patch

    with (
        patch("taxonomic_rag_system.utils.retriever.Chroma"),
        patch("taxonomic_rag_system.utils.retriever.SafeHuggingFaceEmbeddings"),
        patch("taxonomic_rag_system.utils.retriever.RAGChainBuilder"),
    ):
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
async def test_retriever_pipeline():
    """Test the retriever pipeline with a mock RAG model."""
    from unittest.mock import AsyncMock, patch

    from langchain_core.documents import Document

    from taxonomic_rag_system.utils.out_models import TaxBiodiversity

    with (
        patch("taxonomic_rag_system.utils.retriever.Chroma"),
        patch("taxonomic_rag_system.utils.retriever.SafeHuggingFaceEmbeddings"),
        patch("taxonomic_rag_system.utils.retriever.RAGChainBuilder") as mock_builder,
    ):
        # Mock the RAG chain builder
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = TaxBiodiversity(
            classification={
                "Kingdom": "Animalia",
                "Phylum": "N/A",
                "Class": "N/A",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            },
            ancestral="Mock ancestral info",
            specific="Mock specific info",
            commentary="Mock commentary",
            bio_knowledge="Mock bio knowledge",
        )
        mock_builder.return_value = mock_model

        # Create the retriever instance
        retriever = WikiStellaRAGModel(
            vstore_path="/mock/path",
            collection_name="mock_collection",
            embedding_model="mock_model",
        )

        # Mock the retriever's ainvoke method directly to avoid the aretrieve call
        mock_retriever = AsyncMock()
        mock_retriever.ainvoke.return_value = [
            Document(page_content="Mock context", metadata={"source": "test"})
        ]
        retriever.retriever = mock_retriever

        result = await retriever.ainvoke("Mock caption")
        assert result.classification["Kingdom"] == "Animalia"
