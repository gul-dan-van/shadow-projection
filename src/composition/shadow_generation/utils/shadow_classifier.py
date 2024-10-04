import glob
import multiprocessing as mp
import os
import time
import cv2 as cv
import tqdm
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

import skimage.transform as tf
from src.composition.shadow_generation.network.detectron2.config import get_cfg
from src.composition.shadow_generation.network.detectron2.data.detection_utils import read_image
from src.composition.shadow_generation.network.detectron2.utils.logger import setup_logger
from src.composition.shadow_generation.projects.LISA.predictor import VisualizationDemo
from src.composition.utils.model_downloader import ModelDownloader



class ShadowClassifier:
    CONFIG_PATH = "./src/composition/shadow_generation/projects/LISA/config/LISA_101_FPN_3x_demo.yaml"
    def __init__(self, confidence_threshold=0.1):
        self.model_downloader_palette = ModelDownloader('detectron2', './')
        self.model_downloader_palette.download_models()
        self.cfg = self.setup_cfg(confidence_threshold)
        self.model = VisualizationDemo(self.cfg)

    def setup_cfg(self, confidence_threshold):
        cfg = get_cfg()
        cfg.merge_from_file(self.CONFIG_PATH)
        cfg.merge_from_list([])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()

        return cfg

    def predict(self, image, masks=None):
        bounding_boxes = self._get_bounding_boxes(masks)
        
        if len(masks) == 0:
            return 'soft' # Soft Shadow

        union_bbox = self._get_extended_union_bbox(bounding_boxes, image.shape)
        cropped_image = image[union_bbox[1]:union_bbox[3], union_bbox[0]:union_bbox[2]]
        
        cropped_image = image[union_bbox[1]:union_bbox[3], union_bbox[0]:union_bbox[2]]
        try:
            shadow_boxes = self._get_shadow_boxes(cropped_image)
        except:
            return 'soft' # Soft Shadow
        
        adjusted_bounding_boxes = self._adjust_bounding_boxes(bounding_boxes, union_bbox)
        adjusted_shadow_boxes = self._adjust_bounding_boxes(shadow_boxes, [0, 0, 0, 0])

        person_shadow_pairs = self._get_person_shadow_pairs(adjusted_bounding_boxes, adjusted_shadow_boxes)
        pair_ratios = self._calculate_area_ratios(person_shadow_pairs)
        
        return self._classify_shadow(pair_ratios)
    
    def _get_extended_union_bbox(self, bounding_boxes, image_shape):
        if not bounding_boxes:
            return [0, 0, image_shape[1], image_shape[0]]

        x_min = min(box[0] for box in bounding_boxes)
        y_min = min(box[1] for box in bounding_boxes)
        x_max = max(box[2] for box in bounding_boxes)
        y_max = max(box[3] for box in bounding_boxes)

        # Extend the bounding box
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * 0.25))
        x_max = min(image_shape[1], int(x_max + width * 0.25))
        y_max = min(image_shape[0], int(y_max + height * 0.5))

        return [x_min, y_min, x_max, y_max]

    def _adjust_bounding_boxes(self, boxes, reference_bbox):
        adjusted_boxes = []
        for box in boxes:
            adjusted_box = [
                box[0] - reference_bbox[0],
                box[1] - reference_bbox[1],
                box[2] - reference_bbox[0],
                box[3] - reference_bbox[1]
            ]
            adjusted_boxes.append(adjusted_box)
        return adjusted_boxes

    def _get_shadow_boxes(self, image):
        ins = self.model.predictor(image)[0][0]['instances']
        shadow_labels = ins.pred_classes == 1
        return ins.pred_boxes[shadow_labels].tensor.cpu().numpy().astype('float32')

    def _get_bounding_boxes(self, masks):
        bounding_boxes = []
        for mask in masks:
            non_zero = np.nonzero(mask)
            if len(non_zero[0]) > 0 and len(non_zero[1]) > 0:
                y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
                x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
                bounding_boxes.append([x_min, y_min, x_max, y_max])
            else:
                bounding_boxes.append([0,0,0,0])
        return bounding_boxes

    def _get_person_shadow_pairs(self, bounding_boxes, shadow_boxes):
        person_shadow_pairs = []
        for person_bbox in bounding_boxes:
            best_shadow_bbox = self._find_best_shadow_match(person_bbox, shadow_boxes)
            if best_shadow_bbox is not None:
                person_shadow_pairs.append((person_bbox, best_shadow_bbox))
        return person_shadow_pairs

    def _find_best_shadow_match(self, person_bbox, shadow_boxes):
        max_iou = 0
        best_shadow_bbox = None
        person_area = self._calculate_area(person_bbox)
        
        for shadow_bbox in shadow_boxes:
            if shadow_bbox[3] < person_bbox[1] + (person_bbox[3]-person_bbox[1])*.9 \
            or shadow_bbox[1] < person_bbox[1] + (person_bbox[3]-person_bbox[1])*.5:
                continue
            iou = self._calculate_iou(person_bbox, shadow_bbox, person_area)
            if iou > max_iou:
                max_iou = iou
                best_shadow_bbox = shadow_bbox
        
        return best_shadow_bbox

    def _calculate_iou(self, bbox1, bbox2, area1):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area2 = self._calculate_area(bbox2)
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def _calculate_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _calculate_area_ratios(self, person_shadow_pairs):
        pair_ratios = []
        for i, (person_bbox, shadow_bbox) in enumerate(person_shadow_pairs):
            person_area = self._calculate_area(person_bbox)
            shadow_area = self._calculate_area(shadow_bbox)
            area_ratio = shadow_area / person_area if person_area > 0 else 0
            pair_ratios.append(area_ratio)
        return pair_ratios

    def _classify_shadow(self, pair_ratios):
        if not pair_ratios:
            return 'soft'  # Soft Shadow
        return ['soft', 'hard'][int(any(ratio > 0.05 for ratio in pair_ratios))]  
