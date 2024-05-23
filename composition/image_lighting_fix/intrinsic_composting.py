import os 
import argparse
from pathlib import Path
import numpy as np

from glob import glob

from skimage.transform import resize

from chrislib.general import (
    uninvert, 
    np_to_pil, 
    round_32,
)
from chrislib.normal_util import get_omni_normals

from boosted_depth.depth_util import create_depth_models, get_depth

from intrinsic.model_util import load_models
from intrinsic.pipeline import run_pipeline

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    get_light_coeffs
)

from intrinsic_compositing.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

from omnidata_tools.model_util import load_omni_model


def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def compute_composite_normals(img, msk, model, size):
    
    bin_msk = (msk > 0)

    bb = get_bbox(bin_msk)
    bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

    # create the crop around the object in the image to send through normal net
    img_crop = img[bb[0] : bb[1], bb[2] : bb[3], :]

    crop_scale = 1024 / max(bb_h, bb_w)
    img_crop = rescale(img_crop, crop_scale)
        
    # get normals of cropped and scaled object and resize back to original bbox size
    nrm_crop = get_omni_normals(model, img_crop)
    nrm_crop = resize(nrm_crop, (bb_h, bb_w))

    h, w, c = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim
    else:
        scale = 1.0
    
    # resize to the final output size as specified by input 
    out_img = rescale(img, scale, r32=True)
    out_msk = rescale(msk, scale, r32=True)
    out_bin_msk = (out_msk > 0)
    
    # compute normals for the entire composite image at it's output size
    out_nrm_bg = get_omni_normals(model, out_img)
    
    # now the image is at a new size so the parameters of the object crop change.
    # in order to overlay the normals, we need to resize the crop to this new size
    out_bb = get_bbox(out_bin_msk)
    bb_h, bb_w = out_bb[1] - out_bb[0], out_bb[3] - out_bb[2]
    
    # now resize the normals of the crop to this size, and put them in empty image
    out_nrm_crop = resize(nrm_crop, (bb_h, bb_w))
    out_nrm_fg = np.zeros_like(out_img)
    out_nrm_fg[out_bb[0] : out_bb[1], out_bb[2] : out_bb[3], :] = out_nrm_crop

    # combine bg and fg normals with mask alphas
    # print(out_msk.shape)
    out_nrm = (out_nrm_fg * out_msk[:,:,None]) + (out_nrm_bg * (1.0 - out_msk[:,:,None]))
    return out_nrm


class IntrinsicCompositing:
    def __init__(self, config):
        self.config = config
        self.dpt_model = create_depth_models()
        self.nrm_model = load_omni_model()
        self.int_model = load_models('paper_weights')
        self.alb_model = load_albedo_harmonizer()
        self.shd_model = load_reshading_model('further_trained')

    def preprocessing(self,img, bits=8):
        # print(img)
        np_arr = np.array(img).astype(np.single)
        return np_arr / float((2 ** bits) - 1)

    def infer(self, comp, mask, bg, reproduce_paper=False):
        # to ensure that normals are globally accurate we compute them at
        # a resolution of 512 pixels, so resize our shading and image to compute 
        # rescaled normals, then run the lighting model optimization
        # print(comp)

        inference_size = comp.shape[0]
        
        proc_comp = self.preprocessing(comp)
        proc_mask = self.preprocessing(mask)
        proc_bg = self.preprocessing(bg)
        
        bg_h, bg_w = proc_bg.shape[:2]
        max_dim = max(bg_h, bg_w)
        scale = 512 / max_dim
        
        small_bg_img = rescale(proc_bg, scale)
        small_bg_nrm = get_omni_normals(self.nrm_model, small_bg_img)
        
        result = run_pipeline(
            self.int_model,
            small_bg_img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )
        
        small_bg_shd = result['inv_shading'][:, :, None]
        
        
        coeffs, lgt_vis = get_light_coeffs(
            small_bg_shd[:, :, 0], 
            small_bg_nrm, 
            small_bg_img
        )

        # now we compute the normals of the entire composite image, we have some logic
        # to generate a detailed estimation of the foreground object by cropping and 
        # resizing, we then overlay that onto the normals of the whole scene
        comp_nrm = compute_composite_normals(proc_comp, proc_mask, self.nrm_model, inference_size)

        # now compute depth and intrinsics at a specific resolution for the composite image
        # if the image is already smaller than the specified resolution, leave it
        h, w, c = proc_comp.shape
        
        max_dim = max(h, w)
        if max_dim > inference_size:
            scale = inference_size / max_dim
        else:
            scale = 1.0
        
        # resize to specified size and round to 32 for network inference
        img = rescale(proc_comp, scale, r32=True)
        msk = rescale(proc_mask, scale, r32=True)
        
        depth = get_depth(img, self.dpt_model)
        
        result = run_pipeline(
            self.int_model,
            img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )
        
        inv_shd = result['inv_shading']
        # inv_shd = rescale(inv_shd, scale, r32=True)

        # compute the harmonized albedo, and the subsequent color harmonized image
        alb_harm = harmonize_albedo(img, msk, inv_shd, self.alb_model, reproduce_paper=reproduce_paper) ** 2.2
        harm_img = alb_harm * uninvert(inv_shd)[:, :, None]

        # run the reshading model using the various composited components,
        # and our lighting coefficients computed from the background
        comp_result = compute_reshading(
            harm_img,
            msk,
            inv_shd,
            depth,
            comp_nrm,
            alb_harm,
            coeffs,
            self.shd_model
        )
        
        return comp_result['composite'] * 255