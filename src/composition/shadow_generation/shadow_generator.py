
import cv2
import numpy as np
from typing import List

from src.composition.shadow_generation.net.person_segmentation import PersonSegmentationExtractor
from src.composition.shadow_generation.utils.helper import *
from src.composition.shadow_generation.utils.pose_estimation  import get_feet_coords


class ShadowGenerator:
    def __init__(self) -> None:
        self.mask_generator = PersonSegmentationExtractor()

    def transform_masks(self, image, mask, angle, shadow_length, pose_indices=[29, 30, 31, 32]):
        # Ensure the mask is of type uint8
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Apply the mask using cv2.bitwise_and to maintain the data type
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # Proceed with pose estimation
        feet_coords = get_feet_coords(segmented_image, pose_indices=pose_indices)
        if feet_coords is None:
            print("Skipping mask due to no pose detected.")
            return None, None  # Indicate that this mask should be skipped


        foot1_mask = mask.copy()
        foot1_mask[:feet_coords[0][1]] = 0
        foot2_mask = mask.copy()
        foot2_mask[:feet_coords[1][1]] = 0
        feet_mask = np.maximum(foot1_mask, foot2_mask)

        person_mask = mask - feet_mask

        bin_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        x1 = contour[:, :, 0].min()
        x2 = contour[:, :, 0].max()
        y1 = contour[:, :, 1].min()
        y2 = contour[:, :, 1].max()
        bottom_mid = ((x1 + x2) // 2, y2)
        top_mid = ((x1 + x2) // 2, y1)

        translation_angle = (180 + 360 + 45 - angle) % 360
        translation_magnitude = max(1, (y2 - y1) // 120)
        feet_mask = translate_image(feet_mask, translation_angle, translation_magnitude)
        faded_mask = apply_fade_effect(person_mask, bottom_mid)

        tf_top_mid = top_mid
        if 90 < angle % 360 < 270:
            M_flip = flip_mask_vertically(bottom_mid)
            tf_top_mid = apply_transformation_to_point(top_mid, M_flip)

        M_scale = scale_mask_lengthwise(shadow_length, bottom_mid)
        tf_top_mid = apply_transformation_to_point(tf_top_mid, M_scale)

        M_rotate = rotate_mask(angle, bottom_mid)
        tf_top_mid = apply_transformation_to_point(tf_top_mid, M_rotate)

        src = np.array([top_mid, feet_coords[0], feet_coords[1]], dtype=np.float32)
        dst = np.array([tf_top_mid, feet_coords[0], feet_coords[1]], dtype=np.float32)
        H = cv2.getAffineTransform(src, dst)
        height, width = image.shape[:2]
        tf_mask = cv2.warpAffine(person_mask, H, (width, height))

        return tf_mask, feet_mask

    def get_transformed_masks(self, image, masks, angle, shadow_length, pose_indices):
        total_shadow_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        total_feet_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for mask in masks:
            result = self.transform_masks(image, mask, angle, shadow_length, pose_indices)
            if result[0] is None:
                continue  # Skip this mask if no pose was detected

            shadow_mask, feet_mask = result
            total_shadow_mask = np.maximum(total_shadow_mask, shadow_mask)
            total_feet_mask = np.maximum(total_feet_mask, feet_mask)

        return total_shadow_mask, total_feet_mask


    def infer(self, image, blur_size=(15, 15), pose_indices=[29, 30, 31, 32], save_gradient_path=None):
        
        angle = 40          # Angle of the shadow in degrees
        shadow_length = .70   # Length of the shadow
        intensity = 5    # Intensity of the shadow
        angle_offset = 5.0    # Offset for the second shadow in degrees

        masks = self.mask_generator.get_person_masks(image)
        print(len(masks), ": Masks Detected")
        
        combined_mask = np.any([mask > 0 for mask in masks], axis=0)
        combined_mask_3 = np.stack([combined_mask] * 3, axis=-1)

        # Generate shadow masks for both angles
        total_shadow_mask1, total_feet_mask1 = self.get_transformed_masks(image, masks, angle, shadow_length, pose_indices)
        total_shadow_mask2, total_feet_mask2 = self.get_transformed_masks(image, masks, angle + angle_offset, shadow_length, pose_indices)

        # Combine the two shadow masks
        combined_shadow_mask = np.maximum(total_shadow_mask1, total_shadow_mask2)
        combined_feet_mask = np.maximum(total_feet_mask1, total_feet_mask2)

        # Apply the gradient to the combined shadow mask
        shadow_mask_with_gradient = apply_shadow_intensity_gradient(combined_shadow_mask, save_path=save_gradient_path)

        # Blur the shadow mask
        blurred_mask = cv2.GaussianBlur(shadow_mask_with_gradient, blur_size, 0)
        normalized_shadow = (blurred_mask.astype(np.float32) / 255.0) * intensity
        inverted_shadow = 1 - normalized_shadow

        shadowed_image = image.astype(np.float32) * inverted_shadow[..., np.newaxis]
        shadowed_image = np.clip(shadowed_image, 0, 255).astype(np.uint8)

        bin_mask = (combined_feet_mask > 0).astype('uint8')
        contact_shadow_image = add_contact_shadow(bin_mask, shadowed_image, contact_shadow_strength=intensity + 3, blur_size=blur_size[0] * 3)
        cleaned_image = contact_shadow_image * (~combined_mask_3) + image * combined_mask_3

        return cleaned_image