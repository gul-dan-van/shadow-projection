from typing import List

import numpy as np

from src.composition.image_processing.smoothening import BorderSmoothing
from src.composition.shadow_generation.utils.person_segmentation import PersonSegmentationExtractor
from src.composition.shadow_generation.utils.shadow_classifier import ShadowClassifier
from src.composition.shadow_generation.utils.hard_shadow_generator import HardShadowGenerator
from src.composition.shadow_generation.utils.soft_shadow_generator import SoftShadowGenerator
from src.composition.shadow_generation.utils.helper import *


class ShadowGenerator:
    def __init__(self) -> None:
        self.mask_generator = PersonSegmentationExtractor()
        self.shadow_classifier = ShadowClassifier()
        self.hard_shadow_generator = HardShadowGenerator()
        self.soft_shadow_generator = SoftShadowGenerator()
        self.border_smoothing = BorderSmoothing()

        self.shadow_generators = {
            'soft': self.generate_soft_shadow
        }


    def infer_shadow_type(self, composite_image: np.ndarray, masks: List[np.ndarray]) -> str:
        """
        Infer the type of shadow present in the composite image.

        Args:
            composite_image (np.ndarray): The composite image array.

        Returns:
            str: The type of shadow detected.
        """
        # INFER SHADOW TYPE
        return self.shadow_classifier.predict(composite_image, masks)

    def generate_soft_shadow(self, composite_image: np.ndarray, composite_mask: np.ndarray, person_mask: List[np.ndarray]) -> np.ndarray:
        return self.soft_shadow_generator.infer(composite_image, person_mask)

    def infer(self, composite_image: np.ndarray, composite_mask: np.ndarray, background_image: np.ndarray) -> str:
        """
        Infer the type of shadow present in the composite image.

        Args:
            composite_image (np.ndarray): The composite image array.

        Returns:
            str: The type of shadow detected.
        """
        shadowed_image = None
        processed_image = composite_image.astype(np.uint8)
        person_masks = self.mask_generator.get_person_masks(composite_image)
        shadowed_image = self.generate_soft_shadow(processed_image, composite_mask, person_masks)

        for mask in person_masks:
            shadowed_image = self.border_smoothing.infer(shadowed_image, mask, shadowed_image)
        
        return shadowed_image