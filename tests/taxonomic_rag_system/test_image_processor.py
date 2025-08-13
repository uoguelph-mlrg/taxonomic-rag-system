"""Module of unit tests of image processor class in `taxonomic_rag_system` project."""

from unittest.mock import patch

import pytest
from PIL import Image

from taxonomic_rag_system.utils.image_processor import ImageProcessor


def test_process_image_from_url():
    """Test processing an image from a URL."""
    with patch(
        "taxonomic_rag_system.utils.image_processor.imgurl_tob64",
        return_value="mock_b64",
    ) as mock_imgurl:
        result = ImageProcessor.process_image(image_url="http://example.com/image.jpg")
        assert result == "mock_b64"
        mock_imgurl.assert_called_once_with("http://example.com/image.jpg")


def test_process_image_from_file():
    """Test processing an image from a file path."""
    with patch(
        "taxonomic_rag_system.utils.image_processor.imgfile_tob64",
        return_value="mock_file_b64",
    ) as mock_imgfile:
        result = ImageProcessor.process_image(image_path="/path/to/image.jpg")
        assert result == "mock_file_b64"
        mock_imgfile.assert_called_once_with("/path/to/image.jpg")


def test_process_image_from_pil():
    """Test processing a PIL Image object."""
    test_image = Image.new("RGB", (10, 10), color="blue")
    with patch(
        "taxonomic_rag_system.utils.image_processor.pilimg_tob64",
        return_value="mock_pil_b64",
    ) as mock_pilimg:
        result = ImageProcessor.process_image(image_obj=test_image)
        assert result == "mock_pil_b64"
        mock_pilimg.assert_called_once_with(test_image)


def test_process_image_no_input():
    """Test error handling when no image input is provided."""
    with pytest.raises(ValueError, match="Error loading image"):
        ImageProcessor.process_image()
    assert ValueError


def test_process_image_multiple_inputs():
    """Test error handling when multiple image inputs are provided."""
    test_image = Image.new("RGB", (10, 10), color="red")
    with pytest.raises(ValueError, match="Error loading image"):
        ImageProcessor.process_image(
            image_url="http://example.com/image.jpg",
            image_path="/path/to/image.jpg",
            image_obj=test_image,
        )
