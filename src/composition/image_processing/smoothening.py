import cv2
import numpy as np
from types import SimpleNamespace


class BorderSmoothing:
    """
    A class for performing border smoothing on an input image.
    
    Attributes:
        config (SimpleNamespace): Configuration parameters for the border smoothing.
        smoothing_pixels (int): Number of pixels to smooth the border.
        smoothing_radius (int): Radius of the Gaussian blur kernel for smoothing.
    """

    def __init__(self, smoothing_pixels: int = 1, smoothing_radius: int = 1) -> None:
        """
         Args:
            config (SimpleNamespace): Configuration parameters for the border smoothing.
            smoothing_pixels (int): Number of pixels to smooth the border.
            smoothing_radius (int): Radius of the Gaussian blur kernel for smoothing.
        """
        self.smoothing_pixels = smoothing_pixels
        self.smoothing_radius = smoothing_radius
        print('Initializing Border Smoothing Class....')

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Inverts the input mask.

        Args:
            mask (ndarray): Input binary mask.

        Returns:
            ndarray: Inverted binary mask.
        """
        return np.where(mask == 0, 1, 0).astype(np.uint8)

    def extend_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extends the input mask using dilation.

        Args:
            mask (ndarray): Input binary mask.

        Returns:
            ndarray: Extended binary mask.
        """
        kernel = np.ones((1, self.smoothing_pixels * 2 + 1), np.uint8)
        mask_extended = cv2.dilate(mask, kernel, iterations=1)
        return mask_extended

    def convert_3channel(self, inp_mask: np.ndarray) -> np.ndarray:
        """
        Converts a single-channel mask to a 3-channel mask.

        Args:
            inp_mask (ndarray): Single-channel mask.

        Returns:
            ndarray: 3-channel mask.
        """
        return np.repeat(inp_mask[:, :, np.newaxis], 3, axis=2)

    def interpolate_with_smoothing(self, bg_image: np.ndarray, comp_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Interpolates between the background and composite images using smoothing.

        Args:
            bg_image (ndarray): Background image.
            comp_image (ndarray): Composite image.
            mask (ndarray): Binary mask.

        Returns:
            ndarray: Blended image with smoothing applied.
        """
        if bg_image.shape[-1] != comp_image.shape[-1]:
            min_channels = min(bg_image.shape[-1], comp_image.shape[-1])
            bg_image = bg_image[:, :, :min_channels]
            comp_image = comp_image[:, :, :min_channels]

        # mask_near = cv2.dilate(mask, None, iterations=1)
        kernel = np.ones((1, (self.smoothing_pixels) * 2 + 1), np.uint8)
        mask_near = cv2.dilate(mask, kernel, iterations=1)
        blurred_comp_near_mask = cv2.GaussianBlur(comp_image, (3, 3), self.smoothing_radius)
        # blurred_bg_near_mask = cv2.GaussianBlur(bg_image, (0, 0), self.smoothing_radius)
        fin_blurred_near_mask = blurred_comp_near_mask

        blended_image = comp_image.copy()
        blended_image[mask_near != 0] = fin_blurred_near_mask[mask_near != 0]
        return blended_image

    def infer(self, comp: np.ndarray, mask: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """
        Perform border smoothing on the input composite image.

        Args:
            comp (ndarray): Composite image.
            mask (ndarray): Binary mask.
            bg (ndarray): Background image.

        Returns:
            ndarray: Blended image with border smoothing applied.
        """
        mask = mask / float((2 ** 8) - 1)
        inv_mask = self.invert_mask(mask)
        mask_extended = self.extend_mask(mask)
        # inv_mask_extended = self.extend_mask(inv_mask)

        fin_mask = ((mask_extended.astype(np.bool_) & inv_mask.astype(np.bool_)) 
                    # (inv_mask_extended.astype(np.bool_) & mask.astype(np.bool_)) |
                    # (inv_mask.astype(np.bool_) & mask.astype(np.bool_))
                    ).astype(np.uint8)

        blended_image_with_smoothing = self.interpolate_with_smoothing(bg, comp, fin_mask)
        print("Border Smoothening Complete....")
        return blended_image_with_smoothing
