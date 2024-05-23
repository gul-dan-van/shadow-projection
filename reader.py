import cv2
import queue
from threading import Thread, Event
from contextlib import contextmanager
from typing import Dict, Tuple, Union
import numpy as np


class ImageReader:
    """Class for reading images."""

    def __init__(self, image_path: str) -> None:
        """
        Initialize ImageReader.

        Args:
            image_path (str): Path to the image file.
        """
        self.image_path = image_path

    def __read_image(self) -> np.ndarray:
        """Read an image from the specified path."""
        return cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED) 

    def get_image(self) -> np.ndarray:
        """Get the image."""
        if not hasattr(self, 'frame'):
            self.frame = self.__read_image()

        return self.frame

    def get_image_properties(self) -> Dict[str, int]:
        """Get properties of the image."""
        image = self.get_image()

        return {
            'frame_width': image.shape[1],
            'frame_height': image.shape[0]
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
