# Test cases for VideoWriter class
from os.path import join


import pytest
import numpy as np
from utils.writer import VideoWriter

# CREATION OF MOCKING VIDEO READER TO ALLOW TO WRITE VIDEO
OUTPUT_VIDEO_PATH='./output/'

class TestVideoWriter:

    def setup_method(self):
        output_path = join(OUTPUT_VIDEO_PATH, "test_video.mp4")
        video_prop = {'fps': 30.0, 'resolution': (640, 480)}
        self.video_writer = VideoWriter(output_path, video_prop)

    def test_video_writer_initialization(self):
        output_path = "./output/test_video.mp4"
        video_prop = {'fps': 30.0, 'resolution': (640, 480)}
        assert self.video_writer.output_path == output_path
        assert self.video_writer.video_prop == video_prop

    def test_write_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with self.video_writer as writer:
            writer.write_frame(frame)
            assert isinstance(frame, np.ndarray)

    def test_write_frame_with_invalid_data(self):
        frame = "invalid_frame_data"
        with pytest.raises(Exception):
            self.video_writer.write_frame(frame)

    def test_video_writer_exit(self):
        self.video_writer = None
        with pytest.raises(Exception):
            self.video_writer.__exit__(None, None, None)
