import pytest
import cv2
import numpy as np
from time import sleep
from unittest.mock import MagicMock
from utils.reader import VideoReader


class TestVideoReader:
    def setup_method(self):
        self.video_path = "input/video/input_video.mp4"
        self.video_reader = VideoReader(self.video_path)

    def test_read_frames_positive(self):
        sleep(0.05)
        ret, frame = self.video_reader.read_frames()
        assert isinstance(ret, bool)
        assert isinstance(frame, np.ndarray)

    def test_get_video_properties(self):
        properties = self.video_reader.get_video_properties()
        assert isinstance(properties, dict)
        assert 'resolution' in properties
        assert 'fps' in properties
        assert isinstance(properties['resolution'], tuple)
        assert isinstance(properties["fps"], float)

    def test_read_frames_negative(self):
        # Mocking the stream to simulate an error
        with pytest.raises(Exception):
            self.video_reader.stream.read = MagicMock(side_effect=Exception("Mocked error"))
            result = self.video_reader.read_frames()
            assert result is None

    def test_video_reader_initialization(self):
        assert self.video_reader.video_path == self.video_path
        assert self.video_reader.q.qsize() == 0
        assert not self.video_reader.stop_event.is_set()
        assert isinstance(self.video_reader.stream, cv2.VideoCapture)

    def test_read_frames_exception_handling(self):
        with pytest.raises(Exception):
            self.video_reader.stream.read = MagicMock(side_effect=Exception("Mocked error"))
            self.video_reader.read_frames()
