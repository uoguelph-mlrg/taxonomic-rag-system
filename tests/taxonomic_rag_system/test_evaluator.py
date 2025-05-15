"""Module of tests for evaluators and dataloaders in `taxonomic_rag_system` project."""

from unittest.mock import MagicMock

import pytest

from taxonomic_rag_system.utils.evaluator import RareSpeciesEvaluator, process_row


def test_process_row_valid_image():
    """Test processing a valid image row."""
    row = {
        "image": MagicMock(getexif=lambda: None),
        "rarespecies_id": "123",
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Mammalia",
        "order": "Primates",
        "family": "Hominidae",
        "genus": "Homo",
        "sciName": "Homo sapiens",
    }
    result = process_row(row)
    assert result["class_dict"]["Kingdom"] == "Animalia"


@pytest.mark.integration_test()
def test_rarespecies_evaluator_pipeline(mock_docs):
    """Test the RareSpeciesEvaluator pipeline."""
    mock_dataset = MagicMock()
    mock_dataset.img_output_pairs = [
        ("mock_image_1", {"Kingdom": "Animalia", "Phylum": "Chordata"}),
        ("mock_image_2", {"Kingdom": "Plantae", "Phylum": "Tracheophyta"}),
    ]

    evaluator = RareSpeciesEvaluator()
    evaluator.dataset = mock_dataset

    dataloader = evaluator.dataloader(batch_size=1)

    for batch in dataloader:
        images, class_dicts = batch
        assert len(images) == 1
        assert "Kingdom" in class_dicts[0]
