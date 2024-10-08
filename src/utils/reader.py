from os.path import exists
import queue
from threading import Thread, Event
from typing import Dict, Tuple, Union, List
from PIL import Image
from io import BytesIO
import requests  # Add this import

import cv2
import numpy as np
from flask import Request


class ImageReader:
    """
    A class to read images from various sources, including local file paths, Flask requests, and URLs.
    Provides functionality to read single or multiple images concurrently.
    """

    def __init__(self) -> None:
        """Initialize the ImageReader class."""
        pass

    def __read_image_from_path(self, path: str) -> np.ndarray:
        """
        Read an image from the specified file path.

        Parameters:
            - path (str): The path to the image file.

        Returns:
            - np.ndarray: The image read from the file.

        Raises:
            - FileNotFoundError: If the specified path does not exist.
            - ValueError: If the image cannot be read.
        """
        if not exists(path):
            raise RuntimeError(
                f"Image Path '{path}' does not exist. Please check the provided path."
            )
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image from path: {path}")
        return image

    def __read_image_from_stream(self, file_stream: BytesIO) -> np.ndarray:
        """Read an image from a file stream."""
        try:
            file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
            image = cv2.cvtColor(file_bytes, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Failed to decode image from stream")
            return image

        except Exception as e:
            raise ValueError(f"Error reading image from stream: {e}")

    def __read_image_from_url(self, image_url: str) -> np.ndarray:
        """Read an image from a URL."""
        try:
            image_response = requests.get(image_url)
            if image_response.status_code != 200:
                raise ValueError(f"Failed to fetch image from URL: {image_url}")

            image_bytes = image_response.content
            file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError("Failed to decode image from URL")
            return image

        except Exception as e:
            raise ValueError(f"Error reading image from URL: {e}")

    def __read_image(self, source: Union[str, BytesIO], stream_type: str) -> np.ndarray:
        """
        Read an image from a local path, a file stream, or a URL based on the stream type.

        Parameters:
            - source (Union[str, BytesIO]): The source of the image, either a file path, a file stream, or a URL.
            - stream_type (str): The type of the source, either 'local', 'stream', or 'url'.

        Returns:
            - np.ndarray: The image read from the source.

        Raises:
            - ValueError: If the stream type is unsupported.
        """
        stream_methods = {
            'local': self.__read_image_from_path,
            'stream': self.__read_image_from_stream,
            'url': self.__read_image_from_url
        }
        
        if stream_type in stream_methods:
            return stream_methods[stream_type](source)
        else:
            raise RuntimeError(f"Unsupported stream type: {stream_type}")

    def __create_stream(self, file_name: str, file_content: bytes, streams_dict: Dict[str, BytesIO]) -> None:
        """
        Create a stream for a file and add it to the streams dictionary.

        Args:
            file_name (str): The name of the file.
            file_content (bytes): The content of the file.
            streams_dict (Dict[str, BytesIO]): The dictionary to add the new stream to.
        """
        try:
            stream = BytesIO(file_content)
            streams_dict[file_name] = stream
        except Exception as e:
            raise RuntimeError(f"Error creating stream for file '{file_name}': {e}")

    def get_stream_list(self, request: Request, file_names_list: List[str], stream_type: str) -> List[BytesIO]:
        """
        Create a stream for each file in the Flask request, ordered by the provided file names.

        Args:
            request (Request): The Flask request containing the files.
            file_names (List[str]): The list of file names to order the streams.
            stream_type (str): The type of the source, either 'stream' or 'url'.

        Returns:
            List[BytesIO]: A list of BytesIO streams for each file, ordered by the provided file names.
        """
        streams_dict = {}
        threads = []

        if stream_type == 'stream':
            file_items = request.files.items()
        elif stream_type == 'url':
            file_items = request.json.items()
        else:
            raise ValueError(f"Unsupported stream type: {stream_type}")

        try:
            for file_name, file_storage in file_items:
                if file_name in file_names_list:
                    if stream_type == 'stream':
                        print("Getting image data...")
                        file_content = file_storage.read()
                        if file_content is not None or file_content != "":
                            t = Thread(target=self.__create_stream, args=(file_name, file_content, streams_dict))
                        else:
                            raise ValueError("No Image Stream has been created or loaded...")
                    elif stream_type == 'url':
                        print("Getting image urls...")
                        if file_name is not None or file_storage != "":
                            streams_dict[file_name] = file_storage
                            # t = Thread(target=create_stream_from_url, args=(file_name, url_path, streams_dict))
                        else:
                            raise ValueError("No Image Stream has been created or loaded...")
            
            if stream_type == 'stream':
                t.start()
                threads.append(t)
                for t in threads:
                    t.join()

        except Exception as e:
            raise RuntimeError(f"Error creating streams from request: {e}")
        
        # Ensure the streams are ordered by the provided file names
        ordered_streams = [streams_dict[file_name] for file_name in file_names_list if file_name in streams_dict]
        return ordered_streams

    def get_image(self, source: Union[str, BytesIO], stream_type: str) -> np.ndarray:
        """
        Get a single image from the specified source.

        Parameters:
            - source (Union[str, BytesIO]): The source of the image, either a file path or a file stream.
            - stream_type (str): The type of the source, either 'local' or 'stream'.

        Returns:
            - np.ndarray: The image read from the source.

        Raises:
            - TypeError: If the source is neither a string nor a file stream.
        """
        if not isinstance(source, (str, BytesIO)):
            raise TypeError(
                "Source must be a file path (str) or a file stream (BytesIO)"
            )
        try:
            return self.__read_image(source, stream_type)
        except Exception as e:
            raise RuntimeError(f"Error getting image: {e}")

    def get_images(self, source_data: List[Union[str, BytesIO]], stream_type_list: List[str], file_names: List[str]) -> List[np.ndarray]:
        """
        Get multiple images concurrently from the specified sources and maintain the order based on file_names.

        Parameters:
            - source_data (List[Union[str, BytesIO]]): A list of sources, each either a file path or a file stream.
            - stream_type_list (List[str]): A list of stream types, each either 'local' or 'stream'.
            - file_names (List[str]): A list of file names to maintain the order of images.

        Returns:
            - List[np.ndarray]: A list of images read from the sources, ordered by file_names.

        Raises:
            - ValueError: If the lengths of source_data, stream_type_list, and file_names do not match.
            - RuntimeError: If an error occurs while reading the images.
        """
        if len(source_data) != len(stream_type_list) or len(source_data) != len(file_names):
            raise ValueError(
                "source_data, stream_type_list, and file_names must be of the same length"
            )

        images_dict = {}
        threads = []

        def read_image(index: int, source: Union[str, BytesIO], stream_type: str) -> None:
            try:
                image = self.__read_image(source, stream_type)
                images_dict[file_names[index]] = image
            except Exception as e:
                print(f"Error reading image: {e}")
                raise RuntimeError("Error occurred while reading images")

        for index, (source, stream_type) in enumerate(zip(source_data, stream_type_list)):
            t = Thread(target=read_image, args=(index, source, stream_type))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Ensure the images are ordered by the provided file names
        ordered_images = [images_dict[file_name] for file_name in file_names if file_name in images_dict]
        return ordered_images

    def get_image_properties(self, frame: np.ndarray) -> Dict[str, int]:
        """
        Get properties of the image.

        Parameters:
            - frame (np.ndarray): The image for which to get the properties.

        Returns:
            - Dict[str, int]: A dictionary containing the width and height of the image.

        Raises:
            - TypeError: If the frame is not a numpy ndarray.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy ndarray")
        return {
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0]
        }



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
    _, final_image_encoded = cv2.imencode(".jpg", final_image)
    final_image_bytes = final_image_encoded.tobytes()
    final_image_size_mb = len(final_image_bytes) / (1024 * 1024)

    if final_image_size_mb > 5:
        # Compress the image to reduce its size
        pil_image = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        quality = 85  # Initial quality
        while final_image_size_mb > 4.8 and quality > 10:
            buffer.seek(0)
            pil_image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            final_image_bytes = buffer.getvalue()
            final_image_size_mb = len(final_image_bytes) / (1024 * 1024)
            quality -= 5  # Reduce quality in steps
            print(f"Trying compression with quality={quality}, size={final_image_size_mb:.2f}MB")

        buffer.seek(0)
        final_image = np.array(Image.open(buffer).convert("RGB"))
        print("Image compressed to reduce size below 5MB.")

    return final_image