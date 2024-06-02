import cv2
import numpy as np
from io import BytesIO
from typing import Dict, Tuple


def read_image(file_path: str, color_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Reads an image from the specified file path using OpenCV.
    
    Args:
        file_path (str): Path to the image file.
        color_mode (int): OpenCV color mode. Default is cv2.IMREAD_COLOR.
        
    Returns:
        np.ndarray: The image read from the file.
        
    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(file_path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Image at path {file_path} not found.")
    return image


def encode_image_to_bytes(image: np.ndarray) -> BytesIO:
    """
    Encodes a numpy image array to a BytesIO object in JPEG format.
    
    Args:
        image (np.ndarray): Image array to encode.
        
    Returns:
        BytesIO: Encoded image in BytesIO object.
    """
    _, buffer = cv2.imencode('.jpg', image)
    return BytesIO(buffer.tobytes())


def prepare_multipart_data(images_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[BytesIO, str]]:
    """
    Prepares a dictionary of images for multipart/form-data requests.
    
    Args:
        images_dict (Dict[str, np.ndarray]): Dictionary of image arrays.
        
    Returns:
        Dict[str, Tuple[BytesIO, str]]: Dictionary with image names as keys and tuples of encoded image bytes and filenames as values.
    """
    return {name: (encode_image_to_bytes(img), f"{name}.jpg") for name, img in images_dict.items()}
