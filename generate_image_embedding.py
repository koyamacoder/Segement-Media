from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import torch

# checkpoint = "F:\models\sam_vit_h_4b8939.pth"
# model_type = "vit_h"
checkpoint = "F:\models\sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device='cuda')
sam.to(device='cpu')
predictor = SamPredictor(sam)
image = cv2.imread('src/dogs.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("dogs_embedding.npy", image_embedding)
