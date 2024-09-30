import torch
from PIL import Image
import numpy as np

try:
    from app import process_image
except ImportError:
    # We're running as a module
    from .app import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map


# Format conversion helpersdapted from LayerStyle -- but LayerStyle has them wrong; there is no squeeze/unsqueeze
def tensor2pil(t_image: torch.Tensor)  -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)




class F2S2GenerateMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(sam_model_to_config_map.keys())
        model_list.sort()
        return {
            "required": {
                "sam2_model": (model_list, {"default": "sam2_hiera_small.pt"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "subject"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",)
    RETURN_NAMES = ("annotated_image", "mask", "masked_image",)
    FUNCTION = "_process_image"
    CATEGORY = "💃rDancer"

    def _process_image(self, sam2_model: str, image: torch.Tensor, prompt: str = None):
        prompt = prompt.strip() if prompt else ""
        annotated_images, masks, masked_images = [], [], []
        # Convert image from tensor to PIL
        # the image has an extra batch dimension, despite the variable name
        for img in image:
            img = tensor2pil(img).convert("RGB")
            annotated_image, mask, masked_image = process_image(sam2_model, img, prompt)
            annotated_images.append(pil2tensor(annotated_image))
            masks.append(pil2tensor(mask))
            masked_images.append(pil2tensor(masked_image))
        annotated_images = torch.stack(annotated_images)
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        return (annotated_images, masks, masked_images, )


NODE_CLASS_MAPPINGS = {
    "RdancerFlorence2SAM2GenerateMask": F2S2GenerateMask
}

__all__ = ["NODE_CLASS_MAPPINGS"]

if __name__ == "__main__":
    # detect which parameters are filenames -- those are images
    # the rest are prompts
    # call process_image with the images and prompts
    # save the output images
    # return the output images' filenames
    import sys
    import os
    import argparse
    from app import process_image

    # import rdancer_debug # will listen for debugger to attach

    def my_process_image(image_path, prompt):
        from utils.sam import SAM_CHECKPOINT
        image = Image.open(image_path).convert("RGB")
        annotated_image, mask, masked_image = process_image(SAM_CHECKPOINT, image, prompt)
        output_image_path, output_mask_path, output_masked_image_path = f"output_image_{os.path.basename(image_path)}", f"output_mask_{os.path.basename(image_path)}", f"output_masked_image_{os.path.basename(image_path)}"
        annotated_image.save(output_image_path)
        mask.save(output_mask_path)
        masked_image.save(output_masked_image_path)
        return output_image_path, output_mask_path, output_masked_image_path

    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <image_path>[ ...] [<prompt>]")
        sys.exit(1)

    # test which exist as filenames
    images = []
    prompts = []

    for arg in sys.argv[1:]:
        if not os.path.exists(arg):
            prompts.append(arg)
        else:
            images.append(arg)
    
    if len(prompts) > 1:
        raise ValueError("At most one prompt is required")
    if len(images) < 1:
        raise ValueError("At least one image is required")
    
    prompt = prompts[0].strip() if prompts else None

    print(f"Processing {len(images)} image{'' if len(images) == 1 else 's'} with prompt: {prompt}")

    from app import process_image

    for image in images:
        output_image_path, output_mask_path, output_masked_image_path = my_process_image(image, prompt)
    print(f"Saved output image to {output_image_path} and mask to {output_mask_path} and masked image to {output_masked_image_path}")

