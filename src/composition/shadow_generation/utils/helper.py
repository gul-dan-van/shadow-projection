# transform_utils.py

import cv2
import numpy as np

def translate_image(image, angle_degrees, magnitude):
    angle_radians = np.radians(angle_degrees)
    tx = int(magnitude * np.cos(angle_radians))
    ty = int(magnitude * np.sin(angle_radians))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    height, width = image.shape[:2]
    translated_image = cv2.warpAffine(image, M, (width, height))
    return translated_image

def apply_fade_effect(mask, pivot_point):
    fade_mask = np.ones_like(mask, dtype=np.float32)
    y_indices = np.arange(mask.shape[0])
    fade_factors = np.clip(1 - (pivot_point[1] - y_indices) / mask.shape[0], 0, 1)
    fade_mask *= fade_factors[:, np.newaxis]
    faded_mask = (mask.astype(np.float32) / 255.0) * fade_mask
    return (faded_mask * 255).astype(np.uint8)

def apply_transformation_to_point(point, M):
    point_homogeneous = np.array([point[0], point[1], 1]).reshape(3, 1)
    transformed_point = np.dot(M, point_homogeneous)
    return transformed_point[0, 0], transformed_point[1, 0]

def flip_mask_vertically(pivot_point):
    M = np.array([[-1, 0, 2 * pivot_point[0]],
                  [0, 1, 0]], dtype=np.float32)
    return M

def scale_mask_lengthwise(length_factor, pivot_point, width_scale=1.0):
    M = np.array([[width_scale, 0, (1 - width_scale) * pivot_point[0]],
                  [0, length_factor, (1 - length_factor) * pivot_point[1]]])
    return M

def rotate_mask(angle, pivot_point):
    pivot_point = (float(pivot_point[0]), float(pivot_point[1]))
    M = cv2.getRotationMatrix2D(pivot_point, angle, 1)
    return M


def add_contact_shadow(binary_mask: np.ndarray, blended_image: np.ndarray, contact_shadow_strength: float = 0.9, blur_size: int = 75) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    contact_shadow = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
    contact_shadow = cv2.GaussianBlur(contact_shadow, (blur_size, blur_size), sigmaX=5, sigmaY=5)
    contact_shadow = contact_shadow * contact_shadow_strength
    contact_shadow = np.stack([contact_shadow] * 3, axis=-1)
    blended_image = blended_image.astype(np.float32) - (contact_shadow * blended_image.astype(np.float32))
    blended_image = np.clip(blended_image, 0, 255)
    return blended_image.astype(np.uint8)

def apply_shadow_intensity_gradient(shadow_mask: np.ndarray, save_path: str = None) -> np.ndarray:
    h, w = shadow_mask.shape[:2]

    # Vertical gradient (height-based)
    vertical_gradient = np.linspace(0, 1, h).reshape(h, 1)**8
    shadow_with_vertical_gradient = shadow_mask * vertical_gradient

    # Horizontal gradient (center-dark, edges-light)
    horizontal_gradient = np.linspace(1, 1, w)
    horizontal_gradient = np.minimum(horizontal_gradient, horizontal_gradient[::-1])  # Symmetric (darker in center)
    horizontal_gradient = horizontal_gradient.reshape(1, w)
    horizontal_gradient = np.tile(horizontal_gradient, (h, 1))

    # Combine gradients
    shadow_with_both_gradients = shadow_with_vertical_gradient * horizontal_gradient

    # Optionally save the shadow mask for debugging
    if save_path:
        shadow_to_save = np.clip(shadow_with_both_gradients * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, shadow_to_save)

    return shadow_with_both_gradients
