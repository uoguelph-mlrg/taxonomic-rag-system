"""
Module providing utilities for processing images and converting them into base64 format.

Classes:
    ImageProcessor: A class that processes images from various sources (URL, file path,
    or PIL image object) and converts them into base64-encoded strings.

Methods
-------
    process_image(image_url=None, image_obj=None, image_path=None):
        Processes an image from a URL, file path, or PIL image object
        and returns its base64-encoded string.
"""

from .helpers import imgfile_tob64, imgurl_tob64, pilimg_tob64


class ImageProcessor:
    """
    A utility class for processing images from various sources.

    Image sources such as URLs, file paths, or in-memory image objects.
    Static method to convert images into base64-encoded strings.
    """

    def __init__(self):
        pass

    @staticmethod
    def process_image(image_url=None, image_obj=None, image_path=None):
        """
        Process an image and convert it into a base64-encoded string.

        Args:
            image_url (str, optional): The URL of the image to process.
                Defaults to None.
            image_obj (PIL.Image.Image, optional): An in-memory image object
                to process. Defaults to None.
            image_path (str, optional): The file path of the image to process.
                Defaults to None.

        Returns
        -------
            str: The base64-encoded representation of the processed image.

        Raises
        ------
            ValueError: If none or multiple sources are provided for the image.
        """
        if image_obj is None and image_url is not None:
            image_b64 = imgurl_tob64(image_url)
        elif image_obj is not None and image_url is None:
            image_b64 = pilimg_tob64(image_obj)
        elif image_path is not None and image_url is None:
            image_b64 = imgfile_tob64(image_path)
        else:
            raise ValueError("Error loading image")
        return image_b64
