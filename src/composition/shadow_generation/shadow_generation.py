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
            'hard': self.generate_hard_shadow,
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

    def generate_hard_shadow(self, composite_image: np.ndarray, composite_mask: np.ndarray, person_mask: List[np.ndarray]) -> np.ndarray:
        shadowed_image = self.hard_shadow_generator.infer(composite_image, composite_mask)
        
        combined_mask = np.zeros_like(person_mask[0], dtype=np.uint8)
        for mask in person_mask:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        combined_mask = np.any([mask > 0 for mask in person_mask], axis=0)
        combined_mask_3 = np.stack([combined_mask] * 3, axis=-1)

        try:
            # Determine shadow angle based on gradient analysis
            angle = determine_shadow_direction(combined_mask, shadowed_image)
            print(f"Determined shadow angle: {angle} degrees")
        except Exception as e:
            # If there is an error, use default angle
            angle = 15  # Or any default value you prefer
            print(f"Could not determine shadow angle, using default angle {angle} degrees. Error: {e}")

        # Calculate shadow height based on angle
        shadow_height = calculate_shadow_height(angle, 0.65)
        print(f"Shadow height for angle {angle} degrees: {shadow_height}")
        
        angle_offset = 5 
        _, total_feet_mask1 = self.soft_shadow_generator.get_transformed_masks(shadowed_image, person_mask, angle, shadow_height)
        _, total_feet_mask2 = self.soft_shadow_generator.get_transformed_masks(shadowed_image, person_mask, angle + angle_offset, shadow_height)
        combined_feet_mask = np.clip(total_feet_mask1 + total_feet_mask2, 0, 255)
        bin_mask = (combined_feet_mask > 0).astype('uint8')
        contact_shadow_image = add_contact_shadow(bin_mask, shadowed_image, contact_shadow_strength=0.5)
        final_image = contact_shadow_image * (~combined_mask_3) + composite_image * combined_mask_3
        
        return final_image

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
        shadowed_image, shadow_type = None, None
        processed_image = composite_image.astype(np.uint8)
        person_masks = self.mask_generator.get_person_masks(composite_image)
        shadow_type = self.infer_shadow_type(background_image.astype(np.float32), person_masks)
        print(f"Shadow Type: {shadow_type}")
        shadowed_image = self.shadow_generators[shadow_type.lower()](processed_image, composite_mask, person_masks)

        for mask in person_masks:
            shadowed_image = self.border_smoothing.infer(shadowed_image, mask, shadowed_image)
        
        return shadowed_image