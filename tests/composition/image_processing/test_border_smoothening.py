from types import SimpleNamespace
import numpy as np
import pytest
from composition.image_processing.smoothening import BorderSmoothing


class TestBorderSmoothing:
    """
    Test class for the BorderSmoothing class methods.
    """
    @pytest.fixture
    def config(self):
        return SimpleNamespace()

    def setup_class(self):
        """
        Setup method to initialize the BorderSmoothing object.
        """
        self.bs = BorderSmoothing(self.config)

    @classmethod
    def teardown_class(cls):
        """
        Teardown method after all tests in the class have run.
        """
        pass

    def test_invert_mask(self):
        """
        Positive test case for the invert_mask method.
        """
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_output = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        assert np.array_equal(self.bs.invert_mask(mask), expected_output)

    def test_extend_mask(self):
        """
        Positive test case for the extend_mask method.
        """
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected_output = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        assert np.array_equal(self.bs.extend_mask(mask), expected_output)

    def test_convert_3channel(self):
        """
        Positive test case for the convert_3channel method.
        """
        inp_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_output = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [0, 0, 0]]], dtype=np.uint8)
        assert np.array_equal(self.bs.convert_3channel(inp_mask), expected_output)

    def test_invert_mask_negative(self):
        """
        Negative test case for the invert_mask method.
        """
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_output = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        assert not np.array_equal(self.bs.invert_mask(mask), expected_output)

    def test_extend_mask_negative(self):
        """
        Negative test case for the extend_mask method.
        """
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected_output = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        assert not np.array_equal(self.bs.extend_mask(mask), expected_output)

    def test_convert_3channel_negative(self):
        """
        Negative test case for the convert_3channel method.
        """
        inp_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        expected_output = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
        assert not np.array_equal(self.bs.convert_3channel(inp_mask), expected_output)

    def test_interpolate_with_smoothing(self):
        """
        Positive test case for the interpolate_with_smoothing method.
        """
        bg_image = np.zeros((3, 3, 3), dtype=np.uint8)
        comp_image = np.ones((3, 3, 3), dtype=np.uint8)
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        output = self.bs.interpolate_with_smoothing(bg_image, comp_image, mask)
        assert output.shape == (3, 3, 3)

    def test_infer(self):
        """
        Positive test case for the infer method.
        """
        comp = np.ones((3, 3, 3), dtype=np.uint8)
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        bg = np.zeros((3, 3, 3), dtype=np.uint8)
        output = self.bs.infer(comp, mask, bg)
        assert output.shape == (3, 3, 3)

    def test_invalid_input(self):
        """
        Test case to check exception handling for invalid input.
        """
        with pytest.raises(Exception):
            invalid_input = np.array([1, 2, 3])  # Invalid input for the method
            self.bs.some_method(invalid_input)
