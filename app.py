from typing import Tuple, Optional

import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
from utils.sam import load_sam_model

MARKDOWN = """
# Florence2 + SAM2 🔥

This demo integrates Florence2 and SAM2 models for detailed image captioning and object 
detection. Florence2 generates detailed captions that are then used to perform phrase 
grounding. The Segment Anything Model 2 (SAM2) converts these phrase-grounded boxes 
into masks.
"""

EXAMPLES = [
    "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog-3.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog-4.jpeg"
]

DEVICE = torch.device("cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_MODEL = load_sam_model(device=DEVICE)
BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)


def process(
    image_input,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if image_input is None:
        return None, None

    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image_input,
        task=FLORENCE_DETAILED_CAPTION_TASK
    )
    caption = result[FLORENCE_DETAILED_CAPTION_TASK]
    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image_input,
        task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
        text=caption
    )
    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=result,
        resolution_wh=image_input.size
    )
    image = np.array(image_input.convert("RGB"))
    SAM_MODEL.set_image(image)
    mask, score, _ = SAM_MODEL.predict(box=detections.xyxy, multimask_output=False)

    # dirty fix; remove this later
    if len(mask.shape) == 4:
        mask = np.squeeze(mask)

    detections.mask = mask.astype(bool)

    output_image = image_input.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image, caption


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            submit_button_component = gr.Button(value='Submit', variant='primary')

        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image output')
            text_output_component = gr.Textbox(label='Caption output')

    submit_button_component.click(
        fn=process,
        inputs=[image_input_component],
        outputs=[
            image_output_component,
            text_output_component
        ]
    )
    with gr.Row():
        gr.Examples(
            fn=process,
            examples=EXAMPLES,
            inputs=[image_input_component],
            outputs=[
                image_output_component,
                text_output_component
            ],
            run_on_click=True
        )

demo.launch(debug=False, show_error=True, max_threads=1)
