# Adding negative test cases for exception and error handling using pytest:

import pytest
import numpy as np
from types import SimpleNamespace
from src.composition.image_composition import ImageComposition

class TestImageComposition:
    COMPOSITE_ENV_VAR = {
        'model_type': 'PCTNet',
        'model_list': ['border-smoothing', 'harmonization'],
        'debug_mode': False
    }

    IMAGE_ENV_VAR = {
        'model_type': 'PCTNet',
        'model_list': ['blending','border-smoothing', 'harmonization'],
        'debug_mode': False
    }

    def test_process_composite(self):
        config = SimpleNamespace(**self.COMPOSITE_ENV_VAR)
        image_composition = ImageComposition(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8)
        bg_image = np.full((100, 100, 3), 255, dtype=np.uint8)

        processed_frame, processed_mask = image_composition.process_composite(frame, mask, bg_image)

        assert processed_frame.shape == frame.shape
        assert processed_mask.shape == mask.shape

    def test_initialize_models(self):
        config = SimpleNamespace(**self.COMPOSITE_ENV_VAR)
        image_composition = ImageComposition(config)
        model_map = image_composition._ImageComposition__initialize_models()

        assert isinstance(model_map, dict)
        assert all(model_type in model_map for model_type in image_composition.model_list)

    def test_process_composite_invalid_input(self):
        with pytest.raises(Exception):
            config = SimpleNamespace(**self.COMPOSITE_ENV_VAR)
            image_composition = ImageComposition(config)
            frame = None
            mask = np.ones((100, 100), dtype=np.uint8)
            bg_image = np.full((100, 100, 3), 255, dtype=np.uint8)

            # Passing invalid input to trigger an exception
            processed_frame, processed_mask = image_composition.process_composite(None, mask, bg_image)

    def test_initialize_models_invalid_config(self):
        with pytest.raises(Exception):
            # Passing invalid configuration to trigger an exception
            config = SimpleNamespace(model_type='InvalidType', model_list=['invalid_model'], debug_mode=False)
            image_composition = ImageComposition(config)
            model_map = image_composition._ImageComposition__initialize_models()

    # def test_process_image(self):
        #     config = SimpleNamespace(**self.IMAGE_ENV_VAR)
        #     image_composition = ImageComposition(config)
            
        #     fg_image = , dtype=np.uint8)
        #     fg_image[:,:,3] = 255  # Setting alpha channel to fully opaque
        #     bg_image = np.zeros((100, 100, 3), dtype=np.uint8)
        #     bbox = [10, 10, 50, 50]

        #     processed_frame, processed_mask = image_composition.process_image(fg_image, bg_image, bbox)

        #     assert processed_frame.shape == bg_image.shape
        #     assert processed_mask.shape == bg_image.shape[:2]