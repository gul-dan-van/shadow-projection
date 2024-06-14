from os.path import exists
from PIL import Image
from types import SimpleNamespace
import warnings

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms

from composition.image_harmonization.network.pctnet.net import PCTNet 
from composition.image_harmonization.network.white_box.harmonizer import Harmonizer
from composition.utils.model_downloader import ModelDownloader


class ImageHarmonization:
    """
    Class to handle image harmonization using different models like PCTNet and Harmonizer.
    """

    MODEL_PATH = "./composition/image_harmonization/models"

    def __init__(self, config: SimpleNamespace) -> None:
        """
        Initialize the ImageHarmonization class with configuration and model setup.

        Args:
            config (SimpleNamespace): Configuration object with settings.
        """
        # SETTING THE DEVICE 
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True if torch.backends.cudnn.is_available() else False
        warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")

        # SETTING THE MODELS
        self.image_harmonization_models = {
            'PCTNet': PCTNet,
            'Harmonizer': Harmonizer
        }

        # SETTING THE TRANSFORMERS
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
        ])

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
            self.model = self.image_harmonization_models[config.model_type]()

        else:
            raise ValueError("Image Harmonization Model Type does not exist!!...")

        self.model.load_state_dict(self.load_model(weights_path), strict=True)
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
                    self.ema_arguments[i] = ema * ema_argument + (1 - ema) * argument

            harmonized = self.model.restore_image(comp, mask, self.ema_arguments)[-1]

            harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
            harmonized = harmonized.astype('uint8')

        harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
        harmonized = Image.fromarray(harmonized.astype(np.uint8))

        return harmonized

    def get_pct_harmonized_image(self, composite_frame: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        """
        Harmonize an image using the PCTNet model.

        Args:
            composite_frame (np.ndarray): The composite frame array.
            composite_mask (np.ndarray): The mask array for the composite frame.

        Returns:
            np.ndarray: The harmonized image array.
        """
        img = cv2.cvtColor(composite_frame, cv2.COLOR_BGR2RGB)
        mask = composite_mask
        img_lr = cv2.resize(img, (256, 256))
        mask_lr = cv2.resize(composite_mask, (256, 256))

        #to tensor
        img = self.transformer(img).float().to(self.device)
        mask = self.transformer(mask).float().to(self.device)
        img_lr = self.transformer(img_lr).float().to(self.device)
        mask_lr = self.transformer(mask_lr).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img_lr, img, mask_lr, mask)

        if len(outputs.shape) == 4:
            outputs = outputs.squeeze(0)

        outputs = (torch.clamp(255.0 * outputs.permute(1, 2, 0), 0, 255)).detach().cpu().numpy()
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)

        return outputs

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
            'Harmonizer': self.get_whitebox_harmonized_image
        }
        print("Image Harmonization Complete....")
        return image_harmonization_methods[self.config.model_type](composite_frame, composite_mask)