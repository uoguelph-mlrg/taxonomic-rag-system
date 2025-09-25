"""
Module of unit tests for the `taxonomic_rag_system` project.

These tests ensure the correctness
and reliability of the helper functions by validating their behavior with
different inputs and expected outputs.

Tested Functions:
- `format_context`: Formats a list of documents into a context string.
- `format_docs`: Formats a list of documents into a structured string.
- `unique_docs`: Filters out duplicate documents from a list.
- `simple_string_output`: Converts a dictionary into a simple string representation.
- `clean_string_output`: Cleans and formats a dictionary into a structured string.
- `imgurl_tob64`: Converts an image from a URL to a base64-encoded string.
- `imgfile_tob64`: Converts an image file to a base64-encoded string.
- `pilimg_tob64`: Converts a PIL Image object to a base64-encoded string.
- `get_metrics`: Computes evaluation metrics such as accuracy and F1 score.
- `dict_match`: Compares two dictionaries and calculates matching key-value pairs.
- `classify_report`: Generates a classification report for predicted and true dicts.
- `custom_collate_fn`: Custom collate function for batching data in a DataLoader.

Dependencies:
- `PIL.Image` for image processing
- `pytest` for test framework and fixtures from conftest.py
- `unittest.mock` for mocking external dependencies in tests
"""

import os
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pandas as pd
import pytest
from PIL import Image

from taxonomic_rag_system.utils.helpers import (
    classify_report,
    clean_string_output,
    custom_collate_fn,
    dict_match,
    extract_tax_metrics,
    extract_tax_metrics_rs,
    format_context,
    format_docs,
    get_metrics,
    imgfile_tob64,
    imgurl_tob64,
    load_api_keys,
    pilimg_tob64,
    rag_evaluate,
    simple_string_output,
    unique_docs,
    write_overall_metrics,
    write_preds_to_csv,
)


def test_load_api_keys_success():
    """Test successful API key loading."""
    with patch(
        "builtins.open",
        mock_open(read_data="test_key"),
        patch.dict(os.environ, {}, clear=True),
    ):
        load_api_keys()
        assert os.environ["OPENAI_API_KEY"] == "test_key"


def test_load_api_keys_missing_openai():
    """Test error when OpenAI key is missing."""
    with (
        patch(target="builtins.open", side_effect=FileNotFoundError),
        pytest.raises(
            expected_exception=FileNotFoundError, match="Could not find OpenAI API key"
        ),
    ):
        load_api_keys()


def test_write_overall_metrics(tmp_path):
    """Test writing metrics to CSV."""
    rank_metrics = {
        "Kingdom": {"accuracy": 0.95, "f1": 0.93, "count": 10},
        "Phylum": {"accuracy": 0.87, "f1": 0.85, "count": 8},
    }
    overall_metrics = {
        "hp": 0.82,
        "hr": 0.79,
        "hf": 0.80,
    }
    # Create a temporary file for writing CSV
    csv_filename = tmp_path / "test_metrics.csv"

    write_overall_metrics(str(csv_filename), rank_metrics, overall_metrics)

    # Verify file contents
    with open(csv_filename, "r") as f:
        content = f.read()
        assert "Rank,Accuracy,F1,Attempts" in content
        assert "Kingdom,0.9500,0.9300,10" in content
        assert "Phylum,0.8700,0.8500,8" in content
        assert "Hierarchical Precision (hp),0.82" in content
        assert "Hierarchical Recall (hr),0.79" in content
        assert "Hierarchical F1 (hf),0.8" in content


def test_extract_tax_metrics():
    """Test extracting taxonomic metrics from results."""
    result_obj = [
        {
            "true_class": {"Kingdom": "Animalia", "Phylum": "Chordata"},
            "guess_class": {"Kingdom": "Animalia", "Phylum": "Chordata"},
        },
        {
            "true_class": {"Kingdom": "Plantae", "Phylum": "Tracheophyta"},
            "guess_class": {"Kingdom": "Plantae", "Phylum": "Magnoliophyta"},
        },
    ]

    rank_metrics, overall_metrics, guess_classes = extract_tax_metrics(result_obj, verbose=False)

    assert "Kingdom" in rank_metrics
    assert "Phylum" in rank_metrics
    assert len(guess_classes) == 2
    assert guess_classes[0]["Kingdom"] == "Animalia"
    assert rank_metrics["Kingdom"]["count"] == 2.0
    assert rank_metrics["Kingdom"]["accuracy"] == 1.0
    assert "hp" in overall_metrics
    assert "hr" in overall_metrics
    assert "hf" in overall_metrics


def test_extract_tax_metrics_rs():
    """Test extracting taxonomic metrics with RSID enrichment."""
    result_obj = [
        {
            "true_class": {"Kingdom": "Animalia"},
            "guess_class": {"Kingdom": "Animalia"},
            "RSID": "RS001",
        }
    ]

    rank_metrics, overall_metrics, guess_classes = extract_tax_metrics_rs(result_obj, verbose=False)

    assert len(guess_classes) == 1
    assert guess_classes[0]["RSID"] == "RS001"
    assert guess_classes[0]["Kingdom"] == "Animalia"
    assert rank_metrics["Kingdom"]["count"] == 1.0
    assert "hp" in overall_metrics
    assert "hr" in overall_metrics
    assert "hf" in overall_metrics


def test_write_preds_to_csv(tmp_path):
    """Test writing predictions to CSV."""
    guess_classes = [
        {
            "RSID": "RS001",
            "Kingdom": "Animalia",
            "Phylum": "Chordata",
            "Class": "Mammalia",
            "Order": "Primates",
            "Family": "Hominidae",
            "Genus": "Homo",
            "Species": "Homo sapiens",
        }
    ]

    # Create temp file to mock writing to csv
    csv_filename = tmp_path / "test_predictions.csv"

    write_preds_to_csv(guess_classes, str(csv_filename))

    # Verify file contents
    with open(csv_filename, "r") as f:
        content = f.read()
        assert "RSID,Kingdom,Phylum,Class,Order,Family,Genus,Species" in content
        assert (
            "RS001,Animalia,Chordata,Mammalia,Primates,Hominidae,Homo,Homo sapiens"
            in content
        )


@pytest.mark.asyncio
async def test_rag_evaluate():
    """Test RAG evaluation functionality."""
    eval_dict = {
        "caption": "Test caption",
        "context": ["Test context"],
        "guess_class": {"Kingdom": "Animalia"},
        "ancestral": "Test ancestral",
        "specific": "Test specific",
        "commentary": "Test commentary",
        "biodiversity": "Test biodiversity",
    }

    mock_embeddings = MagicMock()

    # Mock the scoring results
    with (
        patch("taxonomic_rag_system.utils.helpers.Faithfulness") as mock_faith,
        patch("taxonomic_rag_system.utils.helpers.ResponseRelevancy") as mock_relevancy,
    ):
        mock_faith_instance = MagicMock()
        mock_relevancy_instance = MagicMock()
        mock_faith.return_value = mock_faith_instance
        mock_relevancy.return_value = mock_relevancy_instance

        mock_faith_instance.single_turn_ascore = AsyncMock(return_value=0.85)
        mock_relevancy_instance.single_turn_ascore = AsyncMock(return_value=0.90)

        result = await rag_evaluate(eval_dict, mock_embeddings)

        assert isinstance(result, pd.DataFrame)
        assert result["faithfulness"].iloc[0] == 0.85
        assert result["response_relevancy"].iloc[0] == 0.90


def test_format_context(mock_docs):
    """Test formatting context from documents."""
    result = format_context(mock_docs)
    assert "Content 1" in result
    assert "Content 2" in result


def test_format_docs(mock_docs):
    """Test formatting documents into a structured string."""
    result = format_docs(mock_docs)
    assert "doc1" in result
    assert "Content 1" in result
    assert "doc2" in result
    assert "Content 2" in result


def test_unique_docs(mock_docs):
    """Test filtering out duplicate documents."""
    docs = [[mock_docs[0], mock_docs[0]]]
    result = unique_docs(docs)
    assert len(result) == 1
    assert result[0].page_content == "Content 1"


def test_simple_string_output():
    """Test converting a dictionary to a simple string."""
    out_dict = {
        "guess_class": {"Kingdom": "Animalia", "Phylum": "Chordata"},
        "ancestral": "Warm-blooded",
        "specific": "Fur",
        "biodiversity": "High",
        "commentary": "Common traits",
    }
    result = simple_string_output(out_dict)
    assert "Animalia" in result
    assert "Warm-blooded" in result
    assert "Fur" in result
    assert "High" in result
    assert "Common traits" in result


def test_clean_string_output():
    """Test cleaning and formatting a dictionary into a structured string."""
    out_dict = {
        "guess_class": {"Kingdom": "Animalia", "Phylum": "Chordata"},
        "caption": "A new species",
        "ancestral": "Vertebrate",
        "specific": "Unique features",
        "commentary": "Interesting discovery",
        "biodiversity": "Rich ecosystem",
    }
    result = clean_string_output(out_dict)
    assert "Animalia" in result
    assert "Chordata" in result
    assert "Vertebrate" in result
    assert "Unique features" in result
    assert "Interesting discovery" in result
    assert "Rich ecosystem" in result


def test_imgurl_tob64(mocker):
    """Test converting an image URL to a base64 string."""
    mocker.patch("requests.get", return_value=mocker.Mock(content=b"image_data"))
    result = imgurl_tob64("http://example.com/image.jpg")
    assert isinstance(result, str)


def test_imgfile_tob64(tmp_path):
    """Test converting an image file to a base64 string."""
    image_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (10, 10), color="blue")
    image.save(image_path)
    result = imgfile_tob64(str(image_path))
    assert isinstance(result, str)


def test_pilimg_tob64():
    """Test converting a PIL Image to a base64 string."""
    image = Image.new("RGB", (10, 10), color="green")
    result = pilimg_tob64(image)
    assert isinstance(result, str)


def test_get_metrics():
    """Test computing evaluation metrics."""
    y_trues = ["Mammalia", "Mammalia"]
    y_preds = ["Mammalia", "Mammalia"]
    result = get_metrics(y_trues, y_preds, "Order")
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


def test_dict_match():
    """Test comparing two dictionaries for matching key-value pairs."""
    y_true_dict = {"A": "1", "B": "2"}
    y_pred_dict = {"A": "1", "B": "3"}
    matches, total = dict_match(y_true_dict, y_pred_dict)
    assert matches == 1
    assert total == 2


def test_classify_report():
    """Test generating a classification report."""
    true_dicts = [
        {"Kingdom": "Animalia", "Phylum": "Arthropoda"},
        {"Kingdom": "Animalia", "Phylum": "Chordata"},
    ]
    pred_dicts = [
        {"Kingdom": "Animalia", "Phylum": "Arthropoda"},
        {"Kingdom": "Animalia", "Phylum": "Mollusca"},
    ]
    rank_metrics, overall_metrics = classify_report(true_dicts, pred_dicts, verbose=False)
    assert "count" in rank_metrics["Kingdom"]
    assert "accuracy" in rank_metrics["Kingdom"]
    assert rank_metrics["Kingdom"]["count"] == 2.0
    assert "hp" in overall_metrics
    assert "hr" in overall_metrics
    assert "hf" in overall_metrics


def test_custom_collate_fn():
    """Test custom collate function."""
    batch = [(1, {"label": "A"}), (2, {"label": "B"})]
    result = custom_collate_fn(batch)
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert result[1][0]["label"] == "A"
