from os.path import exists
import queue
import requests
from threading import Thread, Event
from typing import Dict, Tuple, Union
from io import BytesIO
from PIL import Image
import io

import cv2
import numpy as np
from flask import Request


class ImageReader:
    """Class for reading images."""

    def __init__(self, source: Union[str, Request] = None) -> None:
        """
        Initialize ImageReader.

        Args:
            source (Union[str, Request], optional): Path to the image file or Flask request object. Defaults to None.
        """
        self.frame = None
        if not(isinstance(source, str) or isinstance(source, Request)):
            raise ValueError("Please Enter the source path in the correct format")
        else:
            self.source = source

    def __read_image_from_path(self) -> np.ndarray:
        """Read an image from the specified path."""
        if not exists(self.source):
            raise ValueError("Image Path is incorrect. Please check the provided path...")
        
        return cv2.imread(self.source, cv2.IMREAD_UNCHANGED)

    def __read_image_from_request(self, file_name: str, grayscale: bool) -> np.ndarray:
        """Read an image from a Flask request."""
        if file_name not in self.source.files:
            raise ValueError(f"No '{file_name}' part in the request")
        
        file = self.source.files[file_name]
        if file == '':
            raise ValueError("No selected file")

        image_stream = BytesIO(file.read())
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
 
        if grayscale:
            return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        else:
            return cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    def __read_image_from_url(self, url_path: str, grayscale: bool) -> np.ndarray:
        """Read an image from a GET request containing JSON data."""
        image_response = requests.get(url_path.split('?')[0])
        if image_response.status_code != 200:
            raise ValueError(f"Failed to fetch image from URL: {url_path}")
        
        image_bytes = image_response.content
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        
        if grayscale:
            return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    def get_image_from_path(self) -> np.ndarray:
        """Get the image from a file path."""
        self.frame = self.__read_image_from_path()
        return self.frame

    def get_image_from_request(self, file_name: str, grayscale: bool = False) -> np.ndarray:
        """Get the image from a Flask request."""
        self.frame = self.__read_image_from_request(file_name, grayscale)
        return self.frame

    def get_image_from_url(self, url_path: str, grayscale: bool = False) -> np.ndarray:
        self.frame = self.__read_image_from_url(url_path, grayscale)
        return self.frame

    def get_image_properties(self) -> Dict[str, int]:
        """Get properties of the image."""
        return {
            'frame_width': self.frame.shape[1],
            'frame_height': self.frame.shape[0]
        }

    def generate_urls(self) -> Dict[np.ndarray, np.ndarray]:
        """Generate URLs for background image, foreground mask, and signed URL from the request."""
        return self.source.json


class VideoReader:
    """Class for reading frames from a video stream."""

    def __init__(self, video_path: str) -> None:
        """
        Initialize VideoReader.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.q = queue.Queue(maxsize=1024)
        self.stream = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.stop_event = Event()

        t = Thread(target=self.__read_frames, args=())
        t.daemon = True
        t.start()

    def __read_frames(self) -> None:
        """Read frames from the video stream."""
        try:
            while not self.stop_event.is_set():
                if not self.q.full():
                    ret, frame = self.stream.read()
                    if not ret:
                        self.q.put((ret, frame))
                        break
                    self.q.put((ret, frame))
                else:
                    self.stop_event.wait(timeout=0.1)  # Wait for a short time before trying again

        except Exception as e:
            print(f"Error reading frames: {e}")

        finally:
            self.stream.release()

    def read_frames(self) -> Union[Tuple[bool, np.ndarray], None]:
        """Read frames from the queue."""
        return self.q.get()

    def get_video_properties(self) -> Dict[str, Union[Tuple[int, int], float]]:
        """Get properties of the video."""
        frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            'resolution': (frame_width, frame_height),
            'fps': float(self.stream.get(cv2.CAP_PROP_FPS))
        }


def resize_image(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize the image if its size is larger than 5 MB while maintaining aspect ratio."""
    resize_dimensions = {
        'landscape': (1280, 720),
        'portrait': (720, 1280)
    }
    
    image_size_mb = image.nbytes / (1024 * 1024)
    if image_size_mb > 5.00:
        height, width = image.shape[:2]

        # Determine orientation and set max dimensions accordingly
        if width > height:
            max_width, max_height = resize_dimensions['landscape']
        else:
            max_width, max_height = resize_dimensions['portrait']

        # Calculate the scaling factor
        scale_factor = min(max_width / width, max_height / height)

        # Apply the scaling factor to get new dimensions
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return image

def compress_image(final_image: np.ndarray) -> np.ndarray:
    # Check the size of the final image
    _, final_image_encoded = cv2.imencode(".png", final_image)
    final_image_bytes = final_image_encoded.tobytes()
    final_image_size_mb = len(final_image_bytes) / (1024 * 1024)

    if final_image_size_mb > 5:
        # Compress the image to reduce its size
        pil_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)  # Adjust quality as needed
        buffer.seek(0)
        final_image = np.array(Image.open(buffer).convert("RGB"))
        print("Image compressed to reduce size below 5MB.")

    return final_image
