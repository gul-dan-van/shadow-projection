# border_smoothing.py

import cv2
import numpy as np

class BorderSmoothing:
    def __init__(self, smoothing_pixels: int = 1, smoothing_radius: int = 1) -> None:
        self.smoothing_pixels = smoothing_pixels
        self.smoothing_radius = smoothing_radius

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        return np.where(mask == 0, 1, 0).astype(np.uint8)

    def extend_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((1, self.smoothing_pixels * 2 + 1), np.uint8)
        mask_extended = cv2.dilate(mask, kernel, iterations=1)
        return mask_extended

    def interpolate_with_smoothing(self, bg_image: np.ndarray, comp_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if bg_image.shape[-1] != comp_image.shape[-1]:
            min_channels = min(bg_image.shape[-1], comp_image.shape[-1])
            bg_image = bg_image[:, :, :min_channels]
            comp_image = comp_image[:, :, :min_channels]

        mask_near = cv2.dilate(mask, None, iterations=1)
        blurred_comp_near_mask = cv2.GaussianBlur(comp_image, (3, 3), self.smoothing_radius + 2)
        blurred_bg_near_mask = cv2.GaussianBlur(bg_image, (3, 3), self.smoothing_radius + 2)
        fin_blurred_near_mask = blurred_comp_near_mask / 3 + blurred_bg_near_mask * (2 / 3)

        blended_image = comp_image.copy()
        blended_image[mask_near != 0] = fin_blurred_near_mask[mask_near != 0]
        return blended_image

    def infer(self, comp: np.ndarray, mask: np.ndarray, bg: np.ndarray) -> np.ndarray:
        mask = mask / 255.0  # Normalize mask to [0, 1]
        inv_mask = self.invert_mask(mask)
        mask_extended = self.extend_mask(mask)
        inv_mask_extended = self.extend_mask(inv_mask)

        fin_mask = ((mask_extended.astype(bool) & inv_mask.astype(bool)) |
                    (inv_mask_extended.astype(bool) & mask.astype(bool)) |
                    (inv_mask.astype(bool) & mask.astype(bool))).astype(np.uint8)

        blended_image_with_smoothing = self.interpolate_with_smoothing(bg, comp, fin_mask)
        return blended_image_with_smoothing
