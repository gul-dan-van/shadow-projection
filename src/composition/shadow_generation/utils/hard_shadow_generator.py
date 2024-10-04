import numpy as np
import torch
import warnings
import cv2
from PIL import Image
from functools import partial
from torchvision import transforms
from transformers import CLIPImageProcessor

import src.composition.shadow_generation.network.palette.core.util as Util
from src.composition.shadow_generation.network.palette.net.network import Palette
from src.composition.utils.model_downloader import ModelDownloader


PALETTE_MODEL_CONFIG = {'init_type': 'kaiming', 'module_name': 'custom_diffusion', 'unet': {'in_channels': 7, 'out_channels': 3, "block_out_channels": [
                            64,
                            128,
                            256,
                            256
                        ], "cross_attention_dim": 768}, 'beta_schedule': {'train': {'schedule': 'linear', 'n_timestep': 2000, "linear_start": 1e-06,
                            "linear_end": 0.01}, 'test': {'schedule': 'linear', 'n_timestep': 2000, "linear_start": 1e-06,
                            "linear_end": 0.01}}}

class HardShadowGenerator:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True if torch.backends.cudnn.is_available() else False
        warnings.filterwarnings(
            "ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")
        
        
        self.model_downloader_palette = ModelDownloader('shadow_palette', './')
        self.model_downloader_palette.download_models()

        palette_model_path = './shadow_palette.pth'
        self.palette = Palette(**PALETTE_MODEL_CONFIG)
        self.palette.load_state_dict(self.load_model(palette_model_path), strict=False)
        self.palette.set_new_noise_schedule(phase='test')
        self.palette.to(self.device)
        self.palette.eval()

        self.tfs = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])

        self.tfs_mask = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        self.clip_proc = CLIPImageProcessor()
        self.set_device = partial(Util.set_device, rank=1)

    def load_model(self, model_path: str):
        """
        Load the model weights from a specified path.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            dict: State dictionary of the model.
        """
        return torch.load(model_path, map_location=self.device)

    def infer(self, composite_image: np.ndarray, composite_mask: np.ndarray) -> np.ndarray:
        composite_image_RGB = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        comp_image = self.tfs(Image.fromarray(composite_image_RGB)).unsqueeze(0)
        mask_image = self.tfs_mask(composite_mask).unsqueeze(0)
        orig_mask = np.stack([np.where(np.array(orig_mask.resize((224,224)))>128,1,0)]*3)
        orig_mask = torch.from_numpy(orig_mask)
        comp_image = comp_image*(1-orig_mask)
        cond_image = self.set_device(torch.concat([torch.concat([comp_image, mask_image], axis=1) for _ in range(1)], axis=0), distributed=False)
        bg_img_clip = self.clip_proc(bg_img_clip.resize((224,224)), return_tensors="pt").pixel_values[0].unsqueeze(0)
        bg_img_clip = torch.concat([bg_img_clip for _ in range(1)])
        fin_shd = None

        for _ in range(8):
            output, visuals = self.palette.restoration(cond_image, clip_img_emb=bg_img_clip, sample_num=1)
            mask = Image.fromarray(Util.postprocess(output.cpu())[0]).resize(composite_image.size)
            gray_mask = np.array(mask.convert('L'))
            if np.mean(gray_mask)/255<0.4:
                _, otsu_thresh = cv2.threshold(gray_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_thresh = np.repeat(otsu_thresh[:, :, np.newaxis], 3, axis=2)
                kernel = np.ones((15, 15), np.uint8)
                otsu_thresh_dilated = cv2.dilate(otsu_thresh, kernel, borderType=cv2.BORDER_CONSTANT)
                shadow_area = np.array(mask)*otsu_thresh_dilated
                shadow_area_fp32 = shadow_area.astype(np.float32)
                if fin_shd is None:
                    fin_shd = shadow_area_fp32
                elif np.mean(fin_shd)>np.mean(shadow_area_fp32):
                    fin_shd = shadow_area_fp32
        
        if fin_shd is not None:
            diff = composite_image_RGB.to(np.float32) - fin_shd*2/3
            return cv2.cvtColor(np.where(diff<0,0,diff).astype(np.uint8),cv2.COLOR_RGB2BGR)
        
        return composite_image
            

