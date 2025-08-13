"""
Pytest configuration and fixtures for the taxonomic_rag_system test suite.

This module provides shared fixtures and configuration for all tests in the
taxonomic_rag_system package. It includes mock setups for external dependencies
like API keys and provides standardized test fixtures for common components.

Fixtures:
- `mock_api_keys`: Automatically mocks API keys for OpenAI, OpenRouter, and Cohere
- `mock_captioner`: Mocked DescriptiveCaptioner for image caption generation
- `mock_rag_model`: Mocked WikiStellaRAGModel for RAG pipeline testing
- `mock_image_processor`: Mocked ImageProcessor for image processing operations
- `mock_tax_classifier`: Mocked TaxClassifierVLM for taxonomy classification
- `mock_docs`: Sample document data for testing retrieval operations
- `mock_tax_classifier_pyd`: Alternative mock for Pydantic-based taxonomy classification
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from taxonomic_rag_system.utils.out_models import TaxBiodiversity


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """Automatically mock API keys for all tests."""

    def mock_load_api_keys():
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
        os.environ["COHERE_API_KEY"] = "test-cohere-key"

    monkeypatch.setattr(
        "taxonomic_rag_system.utils.helpers.load_api_keys", mock_load_api_keys
    )
    monkeypatch.setattr(
        "taxonomic_rag_system.core.image_rag.load_api_keys", mock_load_api_keys
    )
    monkeypatch.setattr(
        "taxonomic_rag_system.utils.retriever.load_api_keys", mock_load_api_keys
    )
    # Also set the environment variables directly for any code that checks them
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")


@pytest.fixture
def mock_captioner():
    """Fixture for a mocked captioner."""
    captioner = AsyncMock()
    captioner.generate_caption.return_value = "Mock caption"
    return captioner


@pytest.fixture
def mock_rag_model():
    """Fixture for a mocked RAG model."""
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
