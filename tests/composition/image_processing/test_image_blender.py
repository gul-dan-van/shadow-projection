import numpy as np
import pytest
from types import SimpleNamespace
from composition.image_processing.image_blending import ImageBlending


class TestImageBlending:
    @pytest.fixture
    def config(self):
        return SimpleNamespace()

    @pytest.fixture
    def fg_image(self):
        return np.ones((100, 100, 4), dtype=np.uint8)

    @pytest.fixture
    def bg_img(self):
        return np.zeros((400, 400, 3), dtype=np.uint8)

    @pytest.fixture
    def bbox(self):
        return [50, 50, 150, 150]

    def test_generate_fg_mask_with_alpha_channel(self, fg_image):
        image_blending = ImageBlending(self.config)
        mask = image_blending.generate_fg_mask(fg_image)
        assert mask.shape == (100, 100)

    def test_generate_fg_mask_without_alpha_channel(self, fg_image):
        fg_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_blending = ImageBlending(self.config)
        with pytest.raises(ValueError):
            image_blending.generate_fg_mask(fg_image)

    # def test_infer_composite_image_and_mask(self, fg_image, bg_img, bbox):
    #     image_blending = ImageBlending(self.config)
    #     comp_img, comp_mask = image_blending.infer(fg_image, bg_img, bbox)
    #     assert comp_img.shape == bg_img.shape
    #     assert comp_mask.shape == (200, 200)

    def test_convert_mask_to_bbox(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255
        bbox = ImageBlending.convert_mask_to_bbox(mask)
        print(bbox)
        assert bbox == [30, 20, 70, 80]

    def test_convert_mask_to_bbox_invalid_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            ImageBlending.convert_mask_to_bbox(mask)

    def test_crop_and_resize_foreground(self, fg_image, bbox):
        fg_mask = np.zeros((100, 100), dtype=np.uint8)
        fg_mask[40:60, 40:60] = 255
        fg_region, fg_mask_resized = ImageBlending.crop_and_resize_foreground(fg_image, fg_mask, bbox)
        assert fg_region.shape == (100, 100, 4)
        assert fg_mask_resized.shape == (100, 100)



       
