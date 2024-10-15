import cv2
import pytest
from src.utils.writer import ImageWriter

IMAGE_DATA=cv2.imread('input/composite/composite_frame.jpg')


class TestImageWriter:
    def test_write_image_positive(self):
        # Positive unit test for writing an image successfully
        assert ImageWriter.write_image(IMAGE_DATA, "output", "test_image.jpg") is True

    def test_write_image_negative(self):
        # Negative unit test for writing an image unsuccessfully
        with pytest.raises(ValueError):
            ImageWriter.write_image(None, "output", "test_image.jpg")
