import os
from typing import Union, Any, Tuple, Dict
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

FLORENCE_CHECKPOINT = "microsoft/Florence-2-base"
FLORENCE_OBJECT_DETECTION_TASK = '<OD>'
FLORENCE_DETAILED_CAPTION_TASK = '<MORE_DETAILED_CAPTION>'
FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK = '<CAPTION_TO_PHRASE_GROUNDING>'
FLORENCE_OPEN_VOCABULARY_DETECTION_TASK = '<OPEN_VOCABULARY_DETECTION>'
FLORENCE_DENSE_REGION_CAPTION_TASK = '<DENSE_REGION_CAPTION>'


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def load_florence_model(
    device: torch.device, checkpoint: str = FLORENCE_CHECKPOINT
) -> Tuple[Any, Any]:
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True)
        return model, processor


def run_florence_inference(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image,
    task: str,
    text: str = ""
) -> Tuple[str, Dict]:
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text, task=task, image_size=image.size)
    return generated_text, response
