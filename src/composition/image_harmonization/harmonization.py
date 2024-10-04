from os.path import exists
from PIL import Image
from types import SimpleNamespace
import warnings

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms

from src.composition.image_harmonization.network.pctnet.net import PCTNet
from src.composition.image_harmonization.network.white_box.harmonizer import Harmonizer
from src.composition.image_harmonization.network.palette.net.network import Palette
from src.composition.image_harmonization.network.palette.core.util import postprocess
from src.composition.utils.model_downloader import ModelDownloader

PALETTE_MODEL_CONFIG = {'init_type': 'kaiming', 'module_name': 'guided_diffusion', 'unet': {'in_channel': 6, 'out_channel': 3, 'inner_channel': 64, 'channel_mults': [1, 2, 4, 8], 'attn_res': [16], 'num_head_channels': 32, 'res_blocks': 2, 'dropout': 0.2, 'image_size': 224}, 'beta_schedule': {'train': {'schedule': 'cosine', 'n_timestep': 2000}, 'test': {'schedule': 'cosine', 'n_timestep': 1000}}}


class ImageHarmonization:
    """
    Class to handle image harmonization using different models like PCTNet and Harmonizer.
    """

    MODEL_PATH = "./src/composition/image_harmonization/models"

    def __init__(self, config: SimpleNamespace) -> None:
        """
        Initialize the ImageHarmonization class with configuration and model setup.

        Args:
            config (SimpleNamespace): Configuration object with settings.
        """
        # SETTING THE DEVICE
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True if torch.backends.cudnn.is_available() else False
        warnings.filterwarnings(
            "ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")

        # SETTING THE MODELS
        self.image_harmonization_models = {
            'PCTNet': PCTNet,
            'Harmonizer': Harmonizer,
            'Palette': Palette
        }

        # DOWNLOADING THE MODELS
        if not exists(self.MODEL_PATH):
            self.model_downloader = ModelDownloader(config, self.MODEL_PATH)
            self.model_downloader.download_models()
            weights_path = self.model_downloader.model_path

        else:
            weights_path = f'{self.MODEL_PATH}/{self.config.model_type.lower()}.pth'

        # LOADING THE MODELS
        # if not exists(weights_path):
        #     raise ValueError('Image Harmonizer Model Path does not exist!!')

        if config.model_type in self.image_harmonization_models.keys():
            if config.model_type == 'Palette':
                self.model = self.image_harmonization_models[config.model_type](**PALETTE_MODEL_CONFIG)
            else:
                self.model = self.image_harmonization_models[config.model_type]()

        else:
            raise ValueError(
                "Image Harmonization Model Type does not exist!!...")

        self.model.load_state_dict(self.load_model(weights_path), strict=False)
        if config.model_type == 'Palette':
            self.model.set_new_noise_schedule(phase='test')
        self.model.to(self.device)
        self.model.eval()

        if self.config.model_type == 'Harmonizer':
            self.ema_arguments = None

        print("Initializing Image Harmonization Model....")

    def load_model(self, model_path: str):
        """
        Load the model weights from a specified path.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            dict: State dictionary of the model.
        """
        return torch.load(model_path, map_location=self.device)

    def resize_image(self, image: np.ndarray, size: tuple) -> np.ndarray:
        """Resizes the image to the given size using high-quality interpolation."""
        return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


    def get_whitebox_harmonized_image(self, composite_image: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        """
        Harmonize an image using the whitebox model.

        Args:
            composite_image (np.ndarray): The composite image array.
            composite_mask (np.ndarray): The mask array for the composite image.

        Returns:
            np.ndarray: The harmonized image array.
        """
        ema = 1 - 1 / 30
        # Convert numpy arrays to PIL images
        comp = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        comp = Image.fromarray(comp.astype(np.uint8))

        comp = tf.to_tensor(composite_image)[None, ...].to(self.device)
        mask = tf.to_tensor(composite_mask)[None, ...].to(self.device)

        # Harmonization
        with torch.no_grad():
            arguments = self.model.predict_arguments(comp, mask)

            if self.ema_arguments is None:
                self.ema_arguments = list(arguments)
            else:
                for i, (ema_argument, argument) in enumerate(zip(self.ema_arguments, arguments)):
                    self.ema_arguments[i] = ema * \
                        ema_argument + (1 - ema) * argument

            harmonized = self.model.restore_image(
                comp, mask, self.ema_arguments)[-1]

            harmonized = np.transpose(
                harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
            harmonized = harmonized.astype('uint8')

        harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
        harmonized = Image.fromarray(harmonized.astype(np.uint8))

        return harmonized

    def extract_high_freq(self, original_image: np.ndarray) -> np.ndarray:
        """Extracts high-frequency components from the original image using Gaussian blur and Laplacian filtering."""
        # Apply Gaussian blur to create a smooth version of the original image
        blurred_image = cv2.GaussianBlur(original_image, (25, 25), 0)

        # Subtract the blurred image from the original to extract high-frequency details
        high_freq = cv2.subtract(original_image, blurred_image)

        return high_freq

    def sharpen_image(self, harmonized_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Sharpens the harmonized image by blending it with high-frequency details."""
        # Convert images to float32 to avoid overflow/underflow during calculations
        harmonized_image = harmonized_image.astype(np.float32)
        original_image = original_image.astype(np.float32)

        # Extract high-frequency details from the original image
        high_freq_components = self.extract_high_freq(original_image)

        # Ensure both images are float32 before adding
        high_freq_components = high_freq_components.astype(np.float32)

        # Add the high-frequency details to the harmonized image
        sharpened_image = cv2.add(harmonized_image, high_freq_components)

        # Clip the final result to ensure pixel values are in the [0, 255] range
        sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

        # Smooth the sharpened image with a slight Gaussian blur to reduce sharpness
        smoothed_image = cv2.GaussianBlur(sharpened_image, (5, 5), 1.5)  # Kernel size and sigma control the amount of smoothing

        return smoothed_image

    def get_pct_harmonized_image(self, composite_frame: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        """
        Harmonize an image using the PCTNet model.

        Args:
            composite_frame (np.ndarray): The composite frame array.
            composite_mask (np.ndarray): The mask array for the composite frame.

        Returns:
            np.ndarray: The harmonized image array.
        """
        
        harmonization_model = PCTNet()

        # Load the original mask (in original resolution)
        mask = composite_mask

        # Harmonization is done at 512x512 resolution
        harmonized_result = harmonization_model(composite_frame, mask)

        # Upscale the harmonized result back to the original resolution
        original_size = mask.shape
        harmonized_result_upscaled = self.resize_image(harmonized_result, original_size)

        # Load the original high-resolution blended image
        original_image = composite_frame

        # Sharpen the upscaled harmonized image using the high-frequency details from the original image
        sharpened_image = self.sharpen_image(harmonized_result_upscaled, original_image)

        return sharpened_image


    def get_palette_harmonized_image(self, composite_frame: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        """
        Harmonize an image using the Palette Diffusion model.

        Args:
            composite_frame (np.ndarray): The composite frame array.
            composite_mask (np.ndarray): The mask array for the composite frame.

        Returns:
            np.ndarray: The harmonized image array.
        """
        composite_frame = composite_frame[:, :, :3]
        # PRE PROCESSING STEPS
        tfs_composite = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])

        tfs_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        transformed_frame = tfs_composite(composite_frame).unsqueeze(0).to(self.device)
        transformed_mask = tfs_mask(composite_mask).to(self.device)

        # MODEL INFERENCE
        with torch.no_grad():
            outputs, _ = self.model.restoration(transformed_frame, transformed_frame, y_0=transformed_frame, mask=transformed_mask, sample_num=1)

        # POST PROCESSING STEPS
        edited_image = Image.fromarray(postprocess(outputs.cpu())[0])
        og_comp_image = Image.fromarray(postprocess(transformed_frame.cpu())[0])
        
        # GENERATING HIGH RESOLUTION OUTPUT
        og_comp_image = og_comp_image.resize(Image.fromarray(composite_frame).size)
        edited_image = edited_image.resize(Image.fromarray(composite_frame).size)
        updated_nparray = composite_frame.astype(np.float32) + np.array(edited_image).astype(np.float32) - np.array(og_comp_image).astype(np.float32)
        updated_nparray = np.where(updated_nparray > 255, 255, updated_nparray)
        harmonized_image = np.where(updated_nparray <= 0, 0, updated_nparray)

        # Sharpen the upscaled harmonized image using the high-frequency details from the original image
        harmonized_image = self.sharpen_image(harmonized_image, composite_frame)

        return harmonized_image

    def infer(self, composite_frame: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        """
        Infer the harmonized image based on the model type specified in the configuration.

        Args:
            composite_frame (np.ndarray): The composite frame array.
            composite_mask (np.ndarray): The mask array for the composite frame.

        Returns:
            np.ndarray: The harmonized image array.
        """
        image_harmonization_methods = {
            'PCTNet': self.get_pct_harmonized_image,
            'Harmonizer': self.get_whitebox_harmonized_image,
            'Palette': self.get_palette_harmonized_image
        }
        print("Image Harmonization Complete....")
        return image_harmonization_methods[self.config.model_type](composite_frame, composite_mask)
