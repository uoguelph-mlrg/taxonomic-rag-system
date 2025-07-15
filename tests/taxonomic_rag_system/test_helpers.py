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
- `PIL.Image` for image processing.
- `base64` and `io.BytesIO` for encoding and decoding images.
- `mocker` for mocking external dependencies in tests.
"""

from PIL import Image

from taxonomic_rag_system.utils.helpers import (
    classify_report,
    clean_string_output,
    custom_collate_fn,
    dict_match,
    format_context,
    format_docs,
    get_metrics,
    imgfile_tob64,
    imgurl_tob64,
    pilimg_tob64,
    simple_string_output,
    unique_docs,
)


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
    docs = [mock_docs, mock_docs]
    result = unique_docs(docs)
    assert len(result) == 2
    assert result[0]["page_content"] == "Content 1"


def test_simple_string_output():
    """Test converting a dictionary to a simple string."""
    out_dict = {
        "guess_class": "Mammalia",
        "ancestral": "Warm-blooded",
        "specific": "Fur",
        "biodiversity": "High",
        "commentary": "Common traits",
    }
    result = simple_string_output(out_dict)
    assert "Mammalia" in result
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
    y_trues = ["A", "B", "C"]
    y_preds = ["A", "B", "C"]
    result = get_metrics(y_trues, y_preds, "TestLevel", 3)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


def test_dict_match():
    """Test comparing dictionaries for matching key-value pairs."""
    y_true_dict = {"Kingdom": "Animalia", "Phylum": "Chordata"}
    y_pred_dict = {"Kingdom": "Animalia", "Phylum": "Chordata"}
    correct, total = dict_match(y_true_dict, y_pred_dict)
    assert correct == 2
    assert total == 2


def test_classify_report():
    """Test generating a classification report."""
    true_dicts = [{"Kingdom": "Animalia", "Phylum": "Chordata"}]
    pred_dicts = [{"Kingdom": "Animalia", "Phylum": "Chordata"}]
    result = classify_report(true_dicts, pred_dicts)
    assert "Kingdom" in result
    assert result["Kingdom"]["accuracy"] == 1.0
    assert result["Kingdom"]["f1"] == 1.0


def test_custom_collate_fn():
    """Test custom collate function for batching data."""
    batch = [("image1", {"class": "A"}), ("image2", {"class": "B"})]
    images, class_dicts = custom_collate_fn(batch)
    assert images == ["image1", "image2"]
    assert class_dicts == [{"class": "A"}, {"class": "B"}]
