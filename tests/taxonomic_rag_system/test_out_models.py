"""
Unit tests for Pydantic output models in the taxonomic_rag_system project.

This module validates the structure and validation behavior of Pydantic models
used for structured outputs in the taxonomic classification pipeline.

Tested models:
- `Tax`: Model for taxonomic classification data with hierarchical structure
- `Caption`: Model for image caption generation outputs

The tests ensure proper validation, serialization, and data integrity for
structured outputs used throughout the taxonomic RAG system.
"""

from taxonomic_rag_system.utils.out_models import Caption, Tax


def test_tax_model_validation():
    """Test validation of the Tax model."""
    data = {
        "classification": {
            "Kingdom": "Animalia",
            "Phylum": "Chordata",
            "Class": "Mammalia",
            "Order": "Primates",
            "Family": "Hominidae",
            "Genus": "Homo",
            "Species": "Homo sapiens",
        }
    }
    tax = Tax(**data)
    assert tax.classification["Kingdom"] == "Animalia"


def test_caption_model_validation():
    """Test validation of the Caption model."""
    data = {"caption": "A detailed description of the organism."}
    caption = Caption(**data)
    assert caption.caption == "A detailed description of the organism."
