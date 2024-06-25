from types import SimpleNamespace

import cv2

from src.composition.image_harmonization.harmonization import ImageHarmonization


def main():
    
    IMAGE_PATH="/Users/amritanshupandey/Documents/flam/image-video-blending/co-creation/input/composite/composite_frame.jpg"
    MASK_PATH="/Users/amritanshupandey/Documents/flam/image-video-blending/co-creation/input/composite/composite_mask.jpg"
    
    params = {
        'model_type': 'Palette'
    }

    config = SimpleNamespace(**params)
    composite_frame = cv2.imread(IMAGE_PATH)
    composite_mask = cv2.imread(MASK_PATH)

    image_harmonizer = ImageHarmonization(config)

    final_image = image_harmonizer.get_pallete_harmonized_image(composite_frame, composite_mask)

if __name__ == '__main__':
    main()