"""Image Composition"""
from types import SimpleNamespace

import numpy as np

from writer import ImageWriter
from composition.image_processing.image_blending import ImageBlending
from composition.image_processing.smoothening import BorderSmoothing
from composition.image_harmonization.harmonization import ImageHarmonization



def handle_exceptions(func):
    """
    Decorator to handle exceptions within methods.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None, None
    return wrapper


class ImageComposition:
    """
    Class for image composition operations.

    Attributes:
        config(SimpleNamespace): Configuration object containing parameters.
        model_list(list): List of composition models to apply.
        debug_mode(bool): Boolean indicating whether debug mode is enabled.
        image_writer(ImageWriter): ImageWriter object for writing debug images.
        image_composition_models(dict): Dictionary mapping model names to their classes.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        """
        Initialize the ImageComposition object.

        Args:
            config (SimpleNamespace): Configuration object containing parameters.
        """
        self.config = config
        self.model_list = config.model_list
        self.debug_mode = config.debug_mode

        if self.debug_mode:
            self.image_writer = ImageWriter()

        self.image_composition_models = {
            'blending': ImageBlending,
            'border-smoothing': BorderSmoothing,
            'harmonization': ImageHarmonization,
        }

    @handle_exceptions
    def process_composite(self, frame: np.ndarray, mask: np.ndarray, bg_image: np.ndarray) -> tuple:
        """
        Process image composition using specified models.

        Args:
            frame (ndarray): Foreground image frame.
            mask (ndarray): Mask indicating foreground object.
            bg_image (ndarray): Background image.

        Returns:
            tuple[ndarray, ndarray]: Processed composite frame and mask.
        """
        for model in self.model_list:
            if model in self.image_composition_models:
                print(model)
                model_obj = self.image_composition_models[model](self.config)
                if model in ['border-smoothing']:
                    frame = model_obj.infer(frame, mask, bg_image)
                else:
                    frame = model_obj.infer(frame, mask)

                if self.debug_mode:
                    self.image_writer.write_image(frame, self.config.debug_path, f'{model}.jpg')

        return frame, mask

    @handle_exceptions
    def process_image(self, fg_image: np.ndarray, bg_image: np.ndarray, bbox: list) -> tuple:
        """
        Processing the foreground image on the background based on the coordinate position specified
        by the bounding box coordinates.

        Args:
            fg_image (ndarray): Foreground image.
            bg_image (ndarray): Background image.
            bbox (list): Bounding box of the foreground object.

        Returns:
            tuple[ndarray, ndarray]: Processed composite frame and mask.
        """
        image_blender = self.image_composition_models['blending'](self.config)
        frame, mask = image_blender.infer(fg_image, bg_image, bbox)

        if self.debug_mode:
            self.image_writer.write_image(frame, self.config.debug_path, 'composite_image.jpg')

        frame, mask = self.process_composite(frame, mask, bg_image)

        return frame, mask
