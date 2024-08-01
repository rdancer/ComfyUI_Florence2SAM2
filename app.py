from typing import Tuple, Optional

import gradio as gr
import supervision as sv
import torch
from PIL import Image

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import INFERENCE_MODES, OPEN_VOCABULARY_DETECTION, \
    CAPTION_GROUNDING_MASKS
from utils.sam import load_sam_model, run_sam_inference

MARKDOWN = """
# Florence2 + SAM2 🔥

This demo integrates Florence2 and SAM2 models for detailed image captioning and object 
detection. Florence2 generates detailed captions that are then used to perform phrase 
grounding. The Segment Anything Model 2 (SAM2) converts these phrase-grounded boxes 
into masks.
"""

EXAMPLES = [
    [OPEN_VOCABULARY_DETECTION, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", 'straw'],
    [OPEN_VOCABULARY_DETECTION, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", 'napkin'],
    [OPEN_VOCABULARY_DETECTION, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", 'tail'],
    [CAPTION_GROUNDING_MASKS, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", None],
    [CAPTION_GROUNDING_MASKS, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", None],
]

DEVICE = torch.device("cuda")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_MODEL = load_sam_model(device=DEVICE)
BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#FFFFFF"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


def on_mode_dropdown_change(text):
    return [
        gr.Textbox(visible=text == OPEN_VOCABULARY_DETECTION),
        gr.Textbox(visible=text == CAPTION_GROUNDING_MASKS),
    ]


def process(
    mode_dropdown, image_input, text_input
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not image_input:
        return None, None

    if mode_dropdown == OPEN_VOCABULARY_DETECTION:
        if not text_input:
            return None, None

        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text_input
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_MODEL, image_input, detections)
        return annotate_image(image_input, detections), None

    if mode_dropdown == CAPTION_GROUNDING_MASKS:
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
        detections = run_sam_inference(SAM_MODEL, image_input, detections)
        return annotate_image(image_input, detections), caption


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    mode_dropdown_component = gr.Dropdown(
        choices=INFERENCE_MODES,
        value=INFERENCE_MODES[0],
        label="Mode",
        info="Select a mode to use.",
        interactive=True
    )
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            text_input_component = gr.Textbox(
                label='Text prompt')
            submit_button_component = gr.Button(value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image output')
            text_output_component = gr.Textbox(label='Caption output', visible=False)

    with gr.Row():
        gr.Examples(
            fn=process,
            examples=EXAMPLES,
            inputs=[
                mode_dropdown_component,
                image_input_component,
                text_input_component
            ],
            outputs=[
                image_output_component,
                text_output_component
            ],
            run_on_click=True
        )

    submit_button_component.click(
        fn=process,
        inputs=[
            mode_dropdown_component,
            image_input_component,
            text_input_component
        ],
        outputs=[
            image_output_component,
            text_output_component
        ]
    )
    mode_dropdown_component.change(
        on_mode_dropdown_change,
        inputs=[mode_dropdown_component],
        outputs=[
            text_input_component,
            text_output_component
        ]
    )

demo.launch(debug=False, show_error=True)
