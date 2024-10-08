from typing import List, Tuple

import cv2
import numpy as np

from scipy.spatial import distance

from src.composition.shadow_generation.utils.person_segmentation import PersonSegmentationExtractor
from src.composition.shadow_generation.utils.helper import *
from src.composition.shadow_generation.utils.pose_estimation  import get_feet_coords
from src.composition.image_processing.smoothening import BorderSmoothing


class SoftShadowGenerator:
    CONTACT_SHADOW_STRENGTH = 0.5
    HORIZONTAL_GRADIENT = None
    ANGLE_OFFSET = 5.0
    SHADOW_LENGTH = 0.65
    
    def __init__(self) -> None:
        pass
    def transform_masks(self, image, mask, angle, shadow_length, pose_indices=[29, 30, 31, 32]):
        # Ensure the mask is of type uint8
        height, width = image.shape[:2]
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
        bottom_l = (0, 0)
        bottom_r = (width, 0)

        person_mask = mask - feet_mask

        bin_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        x1 = contour[:, :, 0].min()
        x2 = contour[:, :, 0].max()
        y1 = contour[:, :, 1].min()
        y2 = contour[:, :, 1].max()
        bottom_mid = ((x1 + x2) // 2, y2)
        top_mid = ((x1 + x2) // 2, y1)

        translation_angle = (360 + 90 - angle) % 360
        translation_magnitude = max(1, (y2 - y1) // 120)
        feet_mask = translate_image(feet_mask, translation_angle, translation_magnitude)

        feet_coords, poses = arrange_feet_coords(angle, feet_coords)
        pose_idx = 0
        flag = True
        while flag:
            transformation_list = []
            if 90 < angle % 360 < 270:
                M_flip = flip_mask_vertically(bottom_mid)
                transformation_list.append(M_flip)

            M_scale = scale_mask_lengthwise(shadow_length, bottom_mid)
            transformation_list.append(M_scale)

            M_rotate = rotate_mask(angle, bottom_mid)
            transformation_list.append(M_rotate)

            tf_top_mid = cascading_transformations(top_mid, transformation_list)

            src = np.array([top_mid, feet_coords[poses[pose_idx][0]], feet_coords[poses[pose_idx][1]]], dtype=np.float32)
            dst = np.array([tf_top_mid, feet_coords[poses[pose_idx][0]], feet_coords[poses[pose_idx][1]]], dtype=np.float32)

            triangle_skewness = calculate_angle(*dst.tolist())
            H = cv2.getAffineTransform(src, dst)
            tf_bottom_l = apply_transformation_to_point(bottom_l, H)
            tf_bottom_r = apply_transformation_to_point(bottom_r, H)
            d1 = distance.euclidean(bottom_l, bottom_r)
            d2 = distance.euclidean(tf_bottom_l, tf_bottom_r)

            if (triangle_skewness < 20 or triangle_skewness > 160 or d1/d2<0.8) and pose_idx<len(poses)-1:
                pose_idx += 1
            elif pose_idx>=len(poses)-1:
                pose_idx = 0
                angle += 10
            else:
                flag = False
        H = cv2.getAffineTransform(src, dst)
        tf_mask = cv2.warpAffine(person_mask, H, (width, height))
        # cv2.imwrite("feet_mask.jpg", feet_mask)
        # cv2.imwrite("tf_mask.jpg", tf_mask)
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

    def generate_3d_shadow(self, image, masks, angle, blur_size=(85, 85), shadow_length=0.55, pose_indices=[29, 30, 31, 32], save_gradient_path=None, angle_offset=5):
        combined_mask = np.any([mask > 0 for mask in masks], axis=0)
        combined_mask_3 = np.stack([combined_mask] * 3, axis=-1)

        # Generate shadow masks for both angles
        total_shadow_mask1, total_feet_mask1 = self.get_transformed_masks(image, masks, angle, shadow_length, pose_indices)
        total_shadow_mask2, total_feet_mask2 = self.get_transformed_masks(image, masks, angle + angle_offset, shadow_length, pose_indices)

        shadow_mask_with_gradient1 = apply_shadow_intensity_gradient(total_shadow_mask1, save_path=save_gradient_path)
        shadow_mask_with_gradient2 = apply_shadow_intensity_gradient(total_shadow_mask2, save_path=save_gradient_path)

        # Sum the two shadow masks where they overlap
        shadow_mask_with_gradient = np.clip(shadow_mask_with_gradient1 + shadow_mask_with_gradient2, 0, 255)
        combined_feet_mask = np.clip(total_feet_mask1 + total_feet_mask2, 0, 255)

        # Blur the shadow mask
        blurred_mask = cv2.GaussianBlur(shadow_mask_with_gradient, blur_size, 0)
        normalized_shadow = (blurred_mask.astype(np.float32) / 255.0)
        inverted_shadow = 1 - normalized_shadow

        shadowed_image = image.astype(np.float32) * inverted_shadow[..., np.newaxis]
        shadowed_image = np.clip(shadowed_image, 0, 255).astype(np.uint8)

        bin_mask = (combined_feet_mask > 0).astype('uint8')
        contact_shadow_image = add_contact_shadow(bin_mask, shadowed_image, contact_shadow_strength=self.CONTACT_SHADOW_STRENGTH)

        cleaned_image = contact_shadow_image * (~combined_mask_3) + image * combined_mask_3

        return cleaned_image


    def infer(self, image: np.ndarray, masks: List[np.ndarray], save_gradient_path: str = None):

        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        try:
            # Determine shadow angle based on gradient analysis
            angle = determine_shadow_direction(combined_mask, image)
            print(f"Determined shadow angle: {angle} degrees")
        except Exception as e:
            # If there is an error, use default angle
            angle = 15  # Or any default value you prefer
            print(f"Could not determine shadow angle, using default angle {angle} degrees. Error: {e}")

        # Calculate shadow height based on angle
        shadow_height = calculate_shadow_height(angle, self.SHADOW_LENGTH)
        print(f"Shadow height for angle {angle} degrees: {shadow_height}")

        shadowed_image = self.generate_3d_shadow(image, masks, angle=angle, blur_size=(85, 85), shadow_length=shadow_height, angle_offset=self.ANGLE_OFFSET, save_gradient_path=save_gradient_path)

        return shadowed_image