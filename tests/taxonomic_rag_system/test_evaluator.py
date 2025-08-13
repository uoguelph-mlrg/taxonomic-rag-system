"""Module of tests for evaluators and dataloaders in `taxonomic_rag_system` project."""

from unittest.mock import MagicMock, patch

import pytest

from taxonomic_rag_system.utils.evaluator import RareSpeciesEvaluator, process_row


def test_process_row_valid_image():
    """Test processing a valid image row."""
    row = {
        "file_name": MagicMock(getexif=lambda: None),
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
    assert result["class_dict"]["RSID"] == "123"
    assert result["image"] is not None


def test_process_row_invalid_image():
    """Test processing a row with invalid image row."""
    row = {
        "file_name": "not_an_image_object",  # Invalid image
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
    assert result["image"] is None
    assert result["class_dict"]["RSID"] == ""


@patch("taxonomic_rag_system.utils.evaluator.RareSpeciesImageClassDataset")
def test_rarespecies_evaluator_custom_interval(mock_dataset_class):
    """Test RareSpeciesEvaluator with custom interval."""
    # Mock the dataset class
    mock_dataset_instance = MagicMock()
    mock_dataset_class.return_value = mock_dataset_instance

    # Create evaluator with custom interval
    evaluator = RareSpeciesEvaluator(interval=(50, 99))

    # Verify the dataset class was called with the correct interval
    mock_dataset_class.assert_called_once_with(interval=(50, 99))

    # Verify the dataset was assigned
    assert evaluator.dataset == mock_dataset_instance


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
