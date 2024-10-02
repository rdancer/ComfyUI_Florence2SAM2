from typing import Any
import os

import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

model_to_config_map = {
    # models: sam2_hiera_base_plus.pt  sam2_hiera_large.pt  sam2_hiera_small.pt  sam2_hiera_tiny.pt
    # configs: sam2_hiera_b+.yaml  sam2_hiera_l.yaml  sam2_hiera_s.yaml  sam2_hiera_t.yaml
    "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
    "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
    "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
}
SAM_CHECKPOINT = "sam2_hiera_small.pt"
SAM_CONFIG = "sam2_hiera_s.yaml" # from /usr/local/lib/python3.10/dist-packages/sam2/configs, *not* from either the models directory, or this package's directory

def load_sam_image_model(
    device: torch.device,
    checkpoint: str = SAM_CHECKPOINT,
    config: str = None
) -> SAM2ImagePredictor:
    if config is None:
        config = model_to_config_map[checkpoint]
    model_path = os.path.join("models", "sam2", checkpoint)
    model = build_sam2(config, model_path, device=device)
    return SAM2ImagePredictor(sam_model=model)


def load_sam_video_model(
    device: torch.device,
    config: str = SAM_CONFIG,
    checkpoint: str = SAM_CHECKPOINT
) -> Any:
    return build_sam2_video_predictor(config, checkpoint, device=device)


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
