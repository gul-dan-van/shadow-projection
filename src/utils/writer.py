"""Writer Classes"""
import cv2
import numpy as np 
from os.path import join


class ImageWriter:
    """Class for writing images."""

    @staticmethod
    def write_image(image: np.ndarray, folder_path: str, file_name: str) -> bool:
        """
        Write an image to the specified path.

        Args:
            image: Image data to write.
            image_path (str): Path to write the image.

        Returns:
            bool: True if the image was successfully written, False otherwise.
        """
        if isinstance(image, np.ndarray):
            return cv2.imwrite(join(folder_path, file_name), image)
        else:
            raise ValueError("Please pass your image in a numpy array format....")


class VideoWriter:
    """Class for writing video frames."""

    def __init__(self, output_path: str, video_prop: dict) -> None:
        """
        Initialize VideoWriter.

        Args:
            output_path (str): Path to write the video.
            video_prop (dict): Dictionary containing video properties (e.g., resolution, fps).
        """
        self.output_path = output_path
        self.video_prop = video_prop
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = None

    def __enter__(self):
        """Enter the context manager."""
        try:
            self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.video_prop['fps'], self.video_prop['resolution'])
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}")
        return self

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: Frame data to write.
        """
        if isinstance(frame, np.ndarray):
            self.video_writer.write(frame)
        else:
            raise ValueError('Videowriter can only accept frames in numpy array for')
        
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        try:
            if self.video_writer is not None:
                self.video_writer.release()
        except Exception as e:
            print(f"Error releasing VideoWriter: {e}")
