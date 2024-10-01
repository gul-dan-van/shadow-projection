# segmentation.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.composition.shadow_generation.net.sam2.build_sam import build_sam2
from src.composition.shadow_generation.net.sam2.sam2_image_predictor import SAM2ImagePredictor

class PersonSegmentationExtractor:
    YOLO_MODEL = "./models/yolov10n.pt"
    SAM2_MODEL = "./models/sam2_hiera_small.pt"
    SAM2_CONFIG = "./src/composition/shadow_generation/sam2/sam2_hiera_s.yaml"
    
    def __init__(self, yolo_model_path='yolov10n.pt', sam2_model_path='sam2_hiera_small.pt', sam2_cfg='sam2_hiera_s.yaml'):
        self.yolo_model = YOLO(yolo_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam2_model = build_sam2(sam2_cfg, sam2_model_path, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def get_person_bounding_boxes(self, image):
        outputs = self.yolo_model(image)[0]
        person_bboxes = []
        for i in range(len(outputs.boxes)):
            if int(outputs.boxes.cls[i]) == 0:  # Class 0 is for persons
                bbox = outputs.boxes.xyxy[i].cpu().numpy()
                person_bboxes.append(bbox)
        return person_bboxes

    def get_person_masks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.get_person_bounding_boxes(image)
        person_masks = []
        for bbox in bboxes:
            bbox = np.array(bbox).astype(int)
            self.sam2_predictor.set_image(image_rgb)
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],  # Pass the bounding box to SAM2
                multimask_output=False
            )
            mask = masks.squeeze(0)
            bin_mask = (mask > 0).astype(np.uint8) * 255
            person_masks.append(bin_mask)
        return person_masks
