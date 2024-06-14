import pytest
import numpy as np

from types import SimpleNamespace
from composition.image_harmonization.harmonization import ImageHarmonization

MODEL_PATH="composition/image_harmonization/models"


class TestHarmonization:
    """
    Test class for ImageHarmonization functionality.
    """

    @pytest.fixture(scope='class')
    def config(self):
        """
        Fixture to provide a configuration object.
        
        Returns:
            SimpleNamespace: An empty configuration object.
        """
        env_var = {
            'model_type': 'PCTNet'
        }

        return SimpleNamespace(**env_var)

    @pytest.fixture(scope='class')
    def image_harmonization(self, config):
        """
        Fixture to initialize ImageHarmonization object.
        
        Args:
            config (SimpleNamespace): Configuration object.
        
        Returns:
            ImageHarmonization: An instance of ImageHarmonization.
        """
        return ImageHarmonization(config)

    # def test_load_model(self, image_harmonization):
    #     """
    #     Test method to check the loading of a model.
        
    #     Args:
    #         image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
    #     """
    #     assert isinstance(image_harmonization.load_model(MODEL_PATH), dict)

    def test_load_model_invalid_path(self, image_harmonization):
        """
        Test method to check loading model with an invalid path.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        model_path = "invalid_path.pth"
        with pytest.raises(FileNotFoundError):
            image_harmonization.load_model(model_path)

    # def test_get_whitebox_harmonized_image(self, image_harmonization):
    #     """
    #     Test method to get whitebox harmonized image.
        
    #     Args:
    #         image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
    #     """
    #     composite_image = np.ones((100, 100, 3), dtype=np.uint8)
    #     composite_mask = np.ones((100, 100), dtype=np.uint8)
    #     assert isinstance(image_harmonization.get_whitebox_harmonized_image(composite_image, composite_mask), np.ndarray)

    def test_get_whitebox_harmonized_image_invalid_input(self, image_harmonization):
        """
        Test method to check whitebox harmonized image with invalid input.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        composite_image = "invalid_image_data"
        composite_mask = "invalid_mask_data"
        with pytest.raises(Exception):
            image_harmonization.get_whitebox_harmonized_image(composite_image, composite_mask)

    def test_get_pct_harmonized_image(self, image_harmonization):
        """
        Test method to get percentage harmonized image.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        composite_frame = np.ones((100, 100, 3), dtype=np.uint8)
        composite_mask = np.ones((100, 100), dtype=np.uint8)
        assert isinstance(image_harmonization.get_pct_harmonized_image(composite_frame, composite_mask), np.ndarray)

    def test_get_pct_harmonized_image_invalid_input(self, image_harmonization):
        """
        Test method to check percentage harmonized image with invalid input.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        composite_frame = "invalid_frame_data"
        composite_mask = "invalid_mask_data"
        with pytest.raises(Exception):
            image_harmonization.get_pct_harmonized_image(composite_frame, composite_mask)

    def test_infer(self, image_harmonization):
        """
        Test method to infer harmonized image.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        composite_frame = np.ones((100, 100, 3), dtype=np.uint8)
        composite_mask = np.ones((100, 100), dtype=np.uint8)
        assert isinstance(image_harmonization.infer(composite_frame, composite_mask), np.ndarray)

    def test_infer_invalid_input(self, image_harmonization):
        """
        Test method to check inference with invalid input.
        
        Args:
            image_harmonization (ImageHarmonization): Instance of ImageHarmonization.
        """
        composite_frame = "invalid_frame_data"
        composite_mask = "invalid_mask_data"
        with pytest.raises(Exception):
            image_harmonization.infer(composite_frame, composite_mask)