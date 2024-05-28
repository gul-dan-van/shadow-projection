from os.path import exists
import queue
from threading import Thread, Event
from contextlib import contextmanager
from typing import Dict, Tuple, Union
from io import BytesIO

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
        if file.filename == '':
            raise ValueError("No selected file")
        
        image_stream = BytesIO(file.read())
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
 
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

    def get_image_properties(self) -> Dict[str, int]:
        """Get properties of the image."""
        return {
            'frame_width': self.frame.shape[1],
            'frame_height': self.frame.shape[0]
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

    @contextmanager
    def video_capture(self) -> cv2.VideoCapture:
        """
        Context manager for accessing the video stream.

        Yields:
            cv2.VideoCapture: Video stream object.
        """
        try:

            yield self.stream

        finally:
            self.stop_event.set()
            self.stream.release()