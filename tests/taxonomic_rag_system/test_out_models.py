"""Module for unit tests of various pydantic output models."""

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
