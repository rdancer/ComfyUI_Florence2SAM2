from typing import Tuple

import gradio as gr
import supervision as sv
import torch
from PIL import Image

from utils.florence import load_model, run_inference, FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK

MARKDOWN = """
# Florence-2 + SAM2 🔥
"""

DEVICE = torch.device("cuda")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_model(device=DEVICE)
BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)


def process(
    image_input,
) -> Tuple[Image.Image, str]:
    _, result = run_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image_input,
        task=FLORENCE_DETAILED_CAPTION_TASK
    )
    caption = result[FLORENCE_DETAILED_CAPTION_TASK]
    _, result = run_inference(
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

    output_image = image_input.copy()
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

demo.launch(debug=False, show_error=True, max_threads=1)
