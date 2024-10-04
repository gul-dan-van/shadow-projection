import litserve as ls
import torch
import numpy as np
import warnings
import cv2
from typing import List, Dict

from pydantic import BaseModel

from src.utils.config_manager import ConfigManager
from src.utils.reader import ImageReader, resize_image
from src.composition.image_harmonization.network.pctnet.net import PCTNet
from src.composition.image_harmonization.network.palette.net.network import Palette
from src.composition.image_processing.smoothening import BorderSmoothing
from src.composition.shadow_generation.shadow_generation import ShadowGenerator
from src.composition.utils.model_downloader import ModelDownloader
from base_utils import *


# Define a Pydantic model for input validation
class TextRequestModel(BaseModel):
    process_id: str
    background_img_url: str
    foreground_img_url: str
    params: Dict[str,str]
    output_signed_url: str

PALETTE_MODEL_CONFIG = {'init_type': 'kaiming', 'module_name': 'guided_diffusion', 'unet': {'in_channel': 6, 'out_channel': 3, 'inner_channel': 64, 'channel_mults': [1, 2, 4, 8], 'attn_res': [16], 'num_head_channels': 32, 'res_blocks': 2, 'dropout': 0.2, 'image_size': 224}, 'beta_schedule': {'train': {'schedule': 'cosine', 'n_timestep': 2000}, 'test': {'schedule': 'cosine', 'n_timestep': 1000}}}

# (STEP 1) - DEFINE THE API (compound AI system)
class SimpleLitAPI(ls.LitAPI):
    MODEL_PATH = "./"    

    def setup(self, device):
        print("Downloading and Initializing Co Creation AI Models...")

        self.device = device
        torch.backends.cudnn.enabled = True if torch.backends.cudnn.is_available() else False
        warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")
        
        # SETTING THE MODELS
        self.image_harmonization_models = {
            'pctnet': PCTNet,
            'palette': Palette
        }

        self.foreground_image = None
        self.background_image = None

      
        self.border_smoothing = BorderSmoothing()
        self.shadow_generator = ShadowGenerator()

        self.model_downloader_pctnet = ModelDownloader('pctnet', './')
        self.model_downloader_pctnet.download_models()
        self.model_downloader_palette = ModelDownloader('palette', './')
        self.model_downloader_palette.download_models()

        palette_model_path = './palette.pth'
        self.palette = self.image_harmonization_models['palette'](**PALETTE_MODEL_CONFIG)
        self.palette.load_state_dict(self.load_model(palette_model_path), strict=False)
        self.palette.set_new_noise_schedule(phase='test')
        self.palette.to(self.device)
        self.palette.eval()

        pctnet_model_path = "./pctnet.pth"
        self.pctnet = self.image_harmonization_models['pctnet']()
        self.pctnet.load_state_dict(self.load_model(pctnet_model_path))
        self.pctnet.to(self.device)
        self.pctnet.eval()
        

        print("Initializing Image Harmonization Models....")

    def load_model(self, model_path: str):
        """
        Load the model weights from a specified path.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            dict: State dictionary of the model.
        """
        return torch.load(model_path, map_location=self.device)

    def decode_request(self, request: TextRequestModel, context):
        context["process_id"] = request.process_id
        context["params"] = request.params
        
        context["background_image_url"] = request.background_img_url
        context["foreground_image_url"] = request.foreground_img_url
        context["output_signed_url"] = request.output_signed_url

        image_reader =  ImageReader()
        print("Reading Images....")
        background_image = resize_image(image_reader.get_image(context['background_image_url'], stream_type="url"))
        foreground_image = resize_image(image_reader.get_image(context['foreground_image_url'], stream_type="url"))

        # Ensure the images have the same dimensions
        if foreground_image.shape[:2] != background_image.shape[:2]:
            target_size = (min(foreground_image.shape[1], background_image.shape[1]),
                            min(foreground_image.shape[0], background_image.shape[0]))
            foreground_image = cv2.resize(foreground_image, target_size)
            background_image = cv2.resize(background_image, target_size)

        self.foreground_image = foreground_image
        self.background_image = background_image

        
        composite_frame, composite_mask = simple_blend(foreground_image, background_image.astype(np.uint8))
        print("Created Composite Frame and it's Mask...")
        
        if context['params']["model_type"].lower() == 'pctnet':
            preprocessed_data = preprocess_pctnet(composite_frame, composite_mask, self.device)

        elif context['params']["model_type"].lower() == 'palette':
            preprocessed_data = preprocess_palette(composite_frame, composite_mask, self.device)
        
        else:
            raise ValueError("Model not supported")

        return preprocessed_data

    def predict(self, preprocessed_data: Tuple, context):
    # Easily build compound systems. Run inference and return the output.
        harmonized_image = None
        print("Harmonizing Image....")
        if context['params']['model_type'] == 'pctnet':
            with torch.no_grad():
                composite_frame, composite_mask, img_lr, img, mask_lr, mask = preprocessed_data
                outputs = self.pctnet(img_lr, img, mask_lr, mask)
                harmonized_image = postprocess_pctnet(outputs)
        
        elif context['params']['model_type'] == 'palette':
            with torch.no_grad():
                composite_frame, composite_mask, transformed_frame, transformed_mask = preprocessed_data
                outputs, _ = self.palette.restoration(transformed_frame, transformed_frame, y_0=transformed_frame, mask=transformed_mask, sample_num=1)
                harmonized_image = postprocess_palette(composite_frame, transformed_frame, outputs)
        else:
            raise ValueError("Model type not supported....")

        print("Generating Shadow and Smoothening Borders around the image...")
        final_image = self.shadow_generator.infer(harmonized_image, composite_mask)
        cv2.imwrite('final_image.jpg', final_image)

        return final_image

    def encode_response(self, final_image, context):
        output_url = context['output_signed_url']
        process_id = context['process_id']
        response = 200
        cv2.imwrite("final_image.jpg", final_image)
        gcp_sent_status_code, message = send_image_to_gcp(final_image, output_url)        
        if str(gcp_sent_status_code) != '200':
            response = gcp_sent_status_code
            raise RuntimeError(message)

        # processed_message_status_code, message = send_process_confirmation(process_id)
        # if str(processed_message_status_code) != '200':
        #     response = processed_message_status_code
        #     raise RuntimeError(message)

        return response


# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    # scale with advanced features (batching, GPUs, etc...)
    server = ls.LitServer(SimpleLitAPI(), accelerator="cpu", workers_per_device=1, api_path="/cocreation/predict", timeout=300)
    server.run(port=8000, generate_client_file=False)
