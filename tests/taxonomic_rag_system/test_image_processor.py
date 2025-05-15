"""Module of unit tests of image processor class in `taxonomic_rag_system` project."""

from unittest.mock import patch

from taxonomic_rag_system.utils.image_processor import ImageProcessor


def test_process_image_from_url(mock_image_processor):
    """Test processing an image from a URL."""
    with patch(
        "taxonomic_rag_system.utils.helpers.imgurl_tob64", return_value="mock_b64"
    ) as mock_imgurl:
        result = ImageProcessor.process_image(image_url="http://example.com/image.jpg")
        assert result == "mock_b64"
        mock_imgurl.assert_called_once_with("http://example.com/image.jpg")
