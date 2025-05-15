"""Conftest."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_captioner():
    """Fixture for a mocked captioner."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock(caption="Mock caption")
    return mock_client


@pytest.fixture
def mock_rag_model():
    """Fixture for a mocked RAG model."""
    rag_model = AsyncMock()
    rag_model.ainvoke.return_value = {"classification": {"Kingdom": "Animalia"}}
    rag_model.aretrieve.return_value = [{"page_content": "Mock context"}]
    return rag_model


@pytest.fixture
def mock_image_processor():
    """Fixture for a mocked image processor."""
    image_processor = MagicMock()
    image_processor.process_image.return_value = "mock_image_b64"
    return image_processor


@pytest.fixture
def mock_tax_classifier_dict():
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
