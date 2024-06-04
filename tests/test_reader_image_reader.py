import pytest
from pytest_mock import mocker
import numpy as np
from utils.reader import ImageReader
from tests.helper import *

IMAGE_PATH='input/composite/composite_frame.jpg'

@pytest.fixture(scope='class')
def image_reader_from_path():
    image_reader = ImageReader(source=IMAGE_PATH)
    return image_reader


class TestImageReader:
    
    def test_read_image_from_path_correct_path(self, image_reader_from_path):
        image = image_reader_from_path.get_image_from_path()
        assert isinstance(image, np.ndarray)
    
    def test_read_image_from_path_incorrect_path(self):
        with pytest.raises(ValueError):
            image_reader = ImageReader(source='tests/test_images/incorrect_path.jpg')
            image_reader.get_image_from_path()
    
    # def test_read_image_from_request_valid_file(self, mocker):
    #     # Create a mock request object
    #     request_mock = mocker.Mock()
    #     # Pass the mock request object to ImageReader
    #     image_reader = ImageReader(source=request_mock)
    #     image = image_reader.get_image_from_request(file_name='test_image', grayscale=False)
    #     assert isinstance(image, np.ndarray)

    # def test_read_image_from_request_invalid_file(self, mocker):
    #     # Create a mock request object with empty files
    #     request_mock = mocker.Mock(files={})
    #     # Pass the mock request object to ImageReader
    #     image_reader = ImageReader(source=request_mock)
    #     with pytest.raises(ValueError):
    #         image_reader.get_image_from_request(file_name='test_image', grayscale=False)
    
    def test_get_image_properties(self, image_reader_from_path):
        image_reader_from_path.get_image_from_path()
        properties = image_reader_from_path.get_image_properties()
        assert 'frame_width' in properties
        assert 'frame_height' in properties
