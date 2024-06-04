"""Image Blending Class"""
from types import SimpleNamespace
from typing import Tuple, List

import cv2
import numpy as np


class ImageBlending:
    """
    Class for blending foreground and background images using masks.
    """

    ALPHA_THRESHOLD = 128

    def __init__(self, config: SimpleNamespace) -> None:
        """
        Initialize the ImageBlending class.

        Parameters:
            config (SimpleNamespace): Configuration parameters for the blending process.
        """
        self.config = config
        print('Initializing Image Blending Class....')

    def generate_fg_mask(self, fg_image: np.ndarray) -> np.ndarray:
        """
        Generate foreground mask from the alpha channel of the foreground image.

        Parameters:
            fg_image (np.ndarray): Foreground image in BGR(A) format.

        Returns:
            np.ndarray: Foreground mask.
        """
        if fg_image.shape[2] < 4:
            raise ValueError("Foreground image must have an alpha channel")
        alpha_channel = fg_image[:,:,3]
        mask = np.where(alpha_channel > self.ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
        return mask

    def infer(self, fg_image: np.ndarray, bg_img: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend foreground and background images using the provided bounding box.

        Parameters:
            fg_image (np.ndarray): Foreground image in BGR(A) format.
            bg_img (np.ndarray): Background image in BGR format.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            tuple[np.ndarray, np.ndarray]: Composite image and composite mask.
        """
        fg_mask = self.generate_fg_mask(fg_image)
        fg_image = fg_image[:, :, :3]
        fg_region, fg_mask = self.crop_and_resize_foreground(fg_image, fg_mask, bbox)
        x1, y1, x2, y2 = bbox
        comp_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8)
        comp_mask[y1:y2, x1:x2] = fg_mask
        comp_img = bg_img.copy()
        comp_img[y1:y2, x1:x2] = np.where(fg_mask[:,:,np.newaxis] > self.ALPHA_THRESHOLD, fg_region, comp_img[y1:y2, x1:x2])
        return comp_img, comp_mask

    @staticmethod
    def crop_and_resize_foreground(fg_image: np.ndarray, fg_mask: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop and resize the foreground image and mask based on the provided bounding box.

        Parameters:
            fg_image (np.ndarray): Foreground image in BGR format.
            fg_mask (np.ndarray): Foreground mask.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            tuple[np.ndarray, np.ndarray]: Cropped and resized foreground image and mask.
        """
        x1, y1, x2, y2 = bbox
        fg_bbox = ImageBlending.convert_mask_to_bbox(fg_mask)

        fg_region = fg_image[fg_bbox[1]:fg_bbox[3], fg_bbox[0]:fg_bbox[2]]
        fg_region = cv2.resize(fg_region, (x2 - x1, y2 - y1))

        fg_mask = fg_mask[fg_bbox[1]:fg_bbox[3], fg_bbox[0]:fg_bbox[2]]
        fg_mask = cv2.resize(fg_mask, (x2 - x1, y2 - y1))
        fg_mask = np.where(fg_mask > ImageBlending.ALPHA_THRESHOLD, 255, 0).astype(fg_image.dtype)

        return fg_region, fg_mask

    @staticmethod
    def convert_mask_to_bbox(mask: np.ndarray) -> list:
        """
        Convert a mask to bounding box coordinates.

        Parameters:
            mask (np.ndarray): Mask image.

        Returns:
            list: Bounding box coordinates [x1, y1, x2, y2].
        """
        if mask.ndim == 3:
            mask = mask[..., 0]
        binmask = np.where(mask > ImageBlending.ALPHA_THRESHOLD)
        if not binmask[0].any() or not binmask[1].any():
            raise ValueError("Invalid mask provided")
        x1 = int(np.min(binmask[1]))
        x2 = int(np.max(binmask[1]))
        y1 = int(np.min(binmask[0]))
        y2 = int(np.max(binmask[0]))
        return [x1, y1, x2 + 1, y2 + 1]
