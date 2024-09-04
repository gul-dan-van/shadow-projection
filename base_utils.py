import os
import cv2
import numpy as np
import requests
import torch
import math
from typing import Tuple
from PIL import Image

from torchvision.transforms import transforms
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]

def simple_blend(fg_image: np.ndarray, bg_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg_mask = np.where(fg_image[:, :, 3] > 128, 255, 0)
    blended_image = np.copy(bg_image)
    blended_image[fg_mask != 0] = fg_image[fg_mask != 0]
    return blended_image.astype(np.uint8), fg_mask.astype(np.uint8)


def send_image_to_gcp(image: np.ndarray, signed_url: str) -> Tuple[str, str]:
    """ Uploads an image from a NumPy array to GCS using a pre-signed URL. """
    try:
        # Encode image array to bytes
        _, image_encoded = cv2.imencode(".png", image)
        image_bytes = image_encoded.tobytes()

        headers = {
            'Content-Type': 'image/png'
        }
        print(f"Signed URL Provided: {signed_url}")
        # Perform PUT request to upload the file
        response = requests.put(signed_url, data=image_bytes, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Image uploaded successfully.")
        return response.status_code, "Image uploaded successfully."

    except requests.exceptions.RequestException as e:
        print(f"Error uploading image: {e}")
        return response.status_code, f"Error uploading image: {e}"

def send_process_confirmation(process_id: str) -> Tuple[str, str]:
    # Set environment variable for the environment (e.g., "prod" or "dev")
    env = os.getenv("APP_ENV", "dev")

    try:
        # Construct the URL based on the environment
        url = f"https://zingcam.{env}.flamapp.com/engagement-svc/api/v1/engagements/processed/{process_id}"

        data = {
            "status": "processed"
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.status_code, "processed message sent successfully."
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return response.status_code, f"Error sending processed message: {e}"

def preprocess_pctnet(composite_frame: np.ndarray, composite_mask: np.ndarray, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    print(img.shape)
    print(mask.shape)
    img_lr = cv2.resize(img, (256, 256))
    mask_lr = cv2.resize(composite_mask, (256, 256))

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    # to tensor
    img = transformer(img).float().to(device)
    mask = transformer(mask).float().to(device)
    img_lr = transformer(img_lr).float().to(device)
    mask_lr = transformer(mask_lr).float().to(device)

    return img_lr, img, mask_lr, mask

def preprocess_palette(composite_frame: np.ndarray, composite_mask: np.ndarray, device) -> Tuple[torch.tensor, torch.tensor]:
    """
    Harmonize an image using the PCTNet model.

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

    transformed_frame = tfs_composite(composite_frame).unsqueeze(0).to(device)
    transformed_mask = tfs_mask(composite_mask).to(device)

    return composite_frame, transformed_frame, transformed_mask


def postprocess_pctnet(outputs) -> np.ndarray:
    if len(outputs.shape) == 4:
        outputs = outputs.squeeze(0)

    outputs = (torch.clamp(255.0 * outputs.permute(1, 2, 0),0, 255)).detach().cpu().numpy()
    harmonized_image = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)

    return harmonized_image

def postprocess_palette(composite_frame, transformed_frame, outputs) -> np.ndarray:
    # POST PROCESSING STEPS
    edited_image = Image.fromarray(postprocess(outputs.cpu())[0])
    og_comp_image = Image.fromarray(postprocess(transformed_frame.cpu())[0])
    
    # GENERATING HIGH RESOLUTION OUTPUT
    og_comp_image = og_comp_image.resize(Image.fromarray(composite_frame).size)
    edited_image = edited_image.resize(Image.fromarray(composite_frame).size)
    updated_nparray = composite_frame.astype(np.float32) + np.array(edited_image).astype(np.float32) - np.array(og_comp_image).astype(np.float32)
    updated_nparray = np.where(updated_nparray > 255, 255, updated_nparray)
    harmonized_image = np.where(updated_nparray <= 0, 0, updated_nparray)

    return harmonized_image
