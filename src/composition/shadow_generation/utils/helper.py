# transform_utils.py

import cv2
import numpy as np

from typing import List, Tuple

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


###### SHADOW UTILS #######################################

def add_contact_shadow(binary_mask: np.ndarray, blended_image: np.ndarray, contact_shadow_strength: float = 0.5) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    contact_shadow = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=5)
    contact_shadow = cv2.erode(contact_shadow, np.ones((3, 3), np.uint8), iterations=3)
    
    # contact_shadow = cv2.GaussianBlur(contact_shadow, (105, 105), sigmaX=10, sigmaY=10)
    contact_shadow = contact_shadow * contact_shadow_strength
    contact_shadow = np.stack([contact_shadow] * 3, axis=-1)
    contact_shadow_region = cv2.GaussianBlur(contact_shadow * blended_image.astype(np.float32), (9, 9), sigmaX=10, sigmaY=10)

    blended_image = blended_image.astype(np.float32) - (contact_shadow_region)
    blended_image = np.clip(blended_image, 0, 255)
    # cv2.imwrite("contact_shadow_region.jpg", contact_shadow_region)
    return blended_image.astype(np.uint8)

def apply_shadow_intensity_gradient(shadow_mask: np.ndarray, save_path: str = None, vertical_exp: int = 1, hz_gradient: float = 0.25) -> np.ndarray:
    h, w = shadow_mask.shape[:2]
    
    # Get the bounding box of the foreground object in the mask
    # Find where the mask is non-zero (this corresponds to the foreground object)
    y_indices, x_indices = np.where(shadow_mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        # If no object is detected in the mask, return the original mask
        return shadow_mask

    # Get the bounding box of the foreground object
    top = np.min(y_indices)   # The top of the object
    bottom = np.max(y_indices)  # The bottom of the object
    left = np.min(x_indices)   # The left side of the object
    right = np.max(x_indices)  # The right side of the object

    # Height and width of the object's bounding box
    object_height = bottom - top + 1
    object_width = right - left + 1

    # Vertical gradient applied only within the bounding box
    vertical_gradient = np.linspace(0, 1, object_height).reshape(object_height, 1)**vertical_exp

    # Create a full-size vertical gradient and insert the object's gradient into the mask
    full_vertical_gradient = np.ones((h, w), dtype=np.float32)
    full_vertical_gradient[top:bottom + 1, :] = vertical_gradient

    shadow_with_vertical_gradient = shadow_mask * full_vertical_gradient

    # Horizontal gradient applied only within the bounding box
    horizontal_gradient = np.linspace(hz_gradient, hz_gradient, object_width)
    horizontal_gradient = np.minimum(horizontal_gradient, horizontal_gradient[::-1])  # Symmetric (darker in center)
    horizontal_gradient = horizontal_gradient.reshape(1, object_width)

    # Create a full-size horizontal gradient and insert the object's gradient into the mask
    full_horizontal_gradient = np.ones((h, w), dtype=np.float32)
    full_horizontal_gradient[:, left:right + 1] = np.tile(horizontal_gradient, (h, 1))

    # Combine vertical and horizontal gradients
    shadow_with_both_gradients = shadow_with_vertical_gradient * full_horizontal_gradient*1.5

    # Optionally save the shadow mask for debugging
    if save_path:
        shadow_to_save = np.clip(shadow_with_both_gradients * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, shadow_to_save)

    return shadow_with_both_gradients

def calculate_shadow_height(shadow_angle_degrees, shadow_length):
    """
    Calculate the shadow height given the shadow angle and a constant shadow length.
    The shadow angle is measured from the vertical axis (0 degrees) and increases anticlockwise.

    Args:
        shadow_angle_degrees (float): The angle of the shadow in degrees. 
                                      0 is downward, 90 is horizontal to the left, 
                                      and negative angles go clockwise.

    Returns:
        float: The required shadow height.
    """
    # Convert angle from degrees to radians for the cosine function
    shadow_angle_radians = np.radians(shadow_angle_degrees)
    
    # Calculate the shadow height using the formula H = L * cos(θ)
    shadow_height = shadow_length * np.cos(shadow_angle_radians)
    
    return shadow_height

def get_person_bbox(mask: np.ndarray) -> List[int]:
    """
    Extracts the bounding box coordinates of the person from the foreground mask.

    Args:
        mask (np.ndarray): Binary mask of the person.

    Returns:
        List[int]: Bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    xmin, ymin, xmax, ymax = x, y, x + w, y + h
    return [xmin, ymin, xmax, ymax]

def get_light_gradient_data(composite_frame: np.ndarray, region_bbox_left: List[int], region_bbox_right: List[int]) -> Tuple[float, float]:
    """
    Computes the average gradient magnitudes in the left and right regions of the image.

    Args:
        composite_frame (np.ndarray): The image to analyze.
        region_bbox_left (List[int]): Bounding box [xmin, xmax, ymin, ymax] for the left region.
        region_bbox_right (List[int]): Bounding box [xmin, xmax, ymin, ymax] for the right region.

    Returns:
        Tuple[float, float]: Average gradient magnitudes for the left and right regions.
    """
    xmin_left, xmax_left, ymin_left, ymax_left = region_bbox_left
    xmin_right, xmax_right, ymin_right, ymax_right = region_bbox_right

    # Extract left and right regions based on bounding boxes
    region_left = composite_frame[ymin_left:ymax_left, xmin_left:xmax_left]
    region_right = composite_frame[ymin_right:ymax_right, xmin_right:xmax_right]

    # Convert regions to grayscale
    gray_region_left = cv2.cvtColor(region_left, cv2.COLOR_BGR2GRAY)
    gray_region_right = cv2.cvtColor(region_right, cv2.COLOR_BGR2GRAY)

    # Compute gradients in x and y directions
    grad_x_left = cv2.Sobel(gray_region_left, cv2.CV_64F, 1, 0, ksize=5)
    grad_y_left = cv2.Sobel(gray_region_left, cv2.CV_64F, 0, 1, ksize=5)
    grad_x_right = cv2.Sobel(gray_region_right, cv2.CV_64F, 1, 0, ksize=5)
    grad_y_right = cv2.Sobel(gray_region_right, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitudes
    mag_left = np.sqrt(grad_x_left**2 + grad_y_left**2)
    mag_right = np.sqrt(grad_x_right**2 + grad_y_right**2)

    # Calculate average magnitudes
    avg_mag_left = np.mean(mag_left)
    avg_mag_right = np.mean(mag_right)

    return avg_mag_left, avg_mag_right

def compute_light_angle(avg_mag_left: float, avg_mag_right: float) -> float:
    """
    Determines the light angle based on average gradient magnitudes.

    Args:
        avg_mag_left (float): Average gradient magnitude on the left side.
        avg_mag_right (float): Average gradient magnitude on the right side.

    Returns:
        float: Angle in degrees. Positive for light from the right (shadow left), negative for light from the left (shadow right).
    """
    if avg_mag_left > avg_mag_right:
        return -15  # Light coming from the left, so shadow cast to the right
    else:
        return 15   # Light coming from the right, so shadow cast to the left

def determine_shadow_direction(fg_mask: np.ndarray, blended_image: np.ndarray) -> float:
    """
    Determines the angle at which the shadow should be cast based on image analysis.

    Args:
        fg_mask (np.ndarray): Foreground mask of the person.
        blended_image (np.ndarray): The image to analyze.

    Returns:
        float: The angle at which to cast the shadow.
    """
    # Get the bounding box of the person
    person_bbox = get_person_bbox(fg_mask)
    xmin, ymin, xmax, ymax = person_bbox
    mid_x = (xmin + xmax) // 2

    # Define left and right regions based on the bounding box
    region_bbox_left = [xmin, mid_x, ymin, ymax]
    region_bbox_right = [mid_x, xmax, ymin, ymax]

    # Compute average gradient magnitudes in left and right regions
    avg_mag_left, avg_mag_right = get_light_gradient_data(
        blended_image,
        region_bbox_left,
        region_bbox_right
    )

    # Compute light angle
    angle = compute_light_angle(avg_mag_left, avg_mag_right)

    return angle  # Return angle directly