"""Conftest."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_captioner():
    """Fixture for a mocked captioner."""
    captioner = AsyncMock()
    captioner.generate_caption.return_value = "Mock caption"
    return captioner


@pytest.fixture
def mock_rag_model():
    """Fixture for a mocked RAG model."""
    from langchain_core.documents import Document

    from taxonomic_rag_system.utils.out_models import TaxBiodiversity

    rag_model = AsyncMock()
    rag_model.ainvoke.return_value = TaxBiodiversity(
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
    rag_model.aretrieve.return_value = [
        Document(page_content="Mock context", metadata={"source": "test"})
    ]
    return rag_model


@pytest.fixture
def mock_image_processor():
    """Fixture for a mocked image processor."""
    image_processor = MagicMock()
    image_processor.process_image.return_value = "mock_image_b64"
    return image_processor


@pytest.fixture
def mock_tax_classifier():
    """Fixture for a mocked taxonomic classifier."""
    tax_classifier = AsyncMock()
    tax_classifier.generate_taxonomy.return_value = {
        "Kingdom": "Animalia",
        "Phylum": "Chordata",
    }
    return tax_classifier


@pytest.fixture
def mock_docs():
    """Fixture for mocked documents."""
    return [
        {
            "metadata": {"source": "doc1", "relevance_score": 0.9},
            "page_content": "Content 1",
        },
        {
            "metadata": {"source": "doc2", "relevance_score": 0.8},
            "page_content": "Content 2",
        },
    ]


@pytest.fixture
def mock_tax_classifier_pyd():
    """Fixture for a mocked taxonomic classifier."""
    mock_cap = AsyncMock()
    mock_cap.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"classification": {"Kingdom": "Animalia", "Phylum": "Chordata"}}'
                )
            )
        ]
    )
    return mock_cap
