import pytest
import numpy as np
from types import SimpleNamespace
from composition.image_composition import ImageComposition

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


def test_process_composite():
    config = SimpleNamespace(**COMPOSITE_ENV_VAR)
    image_composition = ImageComposition(config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    bg_image = np.full((100, 100, 3), 255, dtype=np.uint8)

    processed_frame, processed_mask = image_composition.process_composite(frame, mask, bg_image)

    assert processed_frame.shape == frame.shape
    assert processed_mask.shape == mask.shape

def test_process_image():
    config = SimpleNamespace(**IMAGE_ENV_VAR)
    image_composition = ImageComposition(config)
    
    fg_image = np.ones((100, 100, 4), dtype=np.uint8)
    fg_image[:,:,3] = 255  # Setting alpha channel to fully opaque
    bg_image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 50, 50]

    processed_frame, processed_mask = image_composition.process_image(fg_image, bg_image, bbox)

    assert processed_frame.shape == bg_image.size
    assert processed_mask.shape == bg_image.size[:2]

def test_initialize_models():
    config = SimpleNamespace(**COMPOSITE_ENV_VAR)
    image_composition = ImageComposition(config)
    model_map = image_composition._ImageComposition__initialize_models()

    assert isinstance(model_map, dict)
    assert all(model_type in model_map for model_type in image_composition.model_list)