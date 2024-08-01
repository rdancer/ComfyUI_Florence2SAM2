from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"


def load_sam_model(
    device: torch.device,
    config: str = SAM_CONFIG,
    checkpoint: str = SAM_CHECKPOINT
) -> SAM2ImagePredictor:
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(sam_model=model)


def run_sam_inference(
    model: Any,
    image: Image,
    detections: sv.Detections
) -> sv.Detections:
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    mask, score, _ = model.predict(box=detections.xyxy, multimask_output=False)

    # dirty fix; remove this later
    if len(mask.shape) == 4:
        mask = np.squeeze(mask)

    detections.mask = mask.astype(bool)
    return detections
