# from app import process_image

__all__ = ["process_image"]


if __name__ == "__main__":
    # detect which parameters are filenames -- those are images
    # the rest are prompts
    # call process_image with the images and prompts
    # save the output images
    # return the output images' filenames
    import sys
    import os
    import argparse
    from PIL import Image

    def my_process_image(image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        output_image, _ = process_image(image, prompt)
        output_path = f"output_{os.path.basename(image_path)}"
        output_image.save(output_path)
        return output_path

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
        output_path = my_process_image(image, prompt)
    print(f"Output image saved to {output_path}")
else:
    from app import process_image
