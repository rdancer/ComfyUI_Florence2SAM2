import os
from typing import Tuple, Optional

import cv2
# import gradio as gr # Gradio phones home, we don't want that
import numpy as np
# import spaces
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
import gc

import comfy.model_management as mm

try:
    from utils.video import generate_unique_name, create_directory, delete_directory

    from utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK #,
        # FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE #, \
        # IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from utils.sam import load_sam_image_model, run_sam_inference #, load_sam_video_model
except ImportError:
    # We're running as a module
    from .utils.video import generate_unique_name, create_directory, delete_directory

    from .utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK #,
        # FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE #, \
        # IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from .utils.sam import load_sam_image_model, run_sam_inference #, load_sam_video_model

# MARKDOWN = """
# # Florence2 + SAM2 🔥

# <div>
#     <a href="https://github.com/facebookresearch/segment-anything-2">
#         <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block;">
#     </a>
#     <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb">
#         <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block;">
#     </a>
#     <a href="https://blog.roboflow.com/what-is-segment-anything-2/">
#         <img src="https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg" alt="Roboflow" style="display:inline-block;">
#     </a>
#     <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y">
#         <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="display:inline-block;">
#     </a>
# </div>

# This demo integrates Florence2 and SAM2 by creating a two-stage inference pipeline. In 
# the first stage, Florence2 performs tasks such as object detection, open-vocabulary 
# object detection, image captioning, or phrase grounding. In the second stage, SAM2 
# performs object segmentation on the image.
# """

# IMAGE_PROCESSING_EXAMPLES = [
#     [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", 'straw, white napkin, black napkin, hair'],
#     [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", 'tail'],
#     [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", None],
#     [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", None],
# ]
# VIDEO_PROCESSING_EXAMPLES = [
#     ["videos/clip-07-camera-1.mp4", "player in white outfit, player in black outfit, ball, rim"],
#     ["videos/clip-07-camera-2.mp4", "player in white outfit, player in black outfit, ball, rim"],
#     ["videos/clip-07-camera-3.mp4", "player in white outfit, player in black outfit, ball, rim"]
# ]

# VIDEO_SCALE_FACTOR = 0.5
# VIDEO_TARGET_DIRECTORY = "tmp"
# create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = None #torch.device("cuda")
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = None, None
SAM_IMAGE_MODEL = None
# SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


# def on_mode_dropdown_change(text):
#     return [
#         gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
#         gr.Textbox(visible=text == IMAGE_CAPTION_GROUNDING_MASKS_MODE),
#     ]

def lazy_load_models(device: torch.device, sam_image_model: str):
    global SAM_IMAGE_MODEL
    global loaded_sam_image_model
    global FLORENCE_MODEL
    global FLORENCE_PROCESSOR
    global DEVICE
    if device != DEVICE:
        offload_models(delete=True)
        DEVICE = device
    if SAM_IMAGE_MODEL is None: 
        SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE, checkpoint=sam_image_model)
        loaded_sam_image_model = sam_image_model
    elif loaded_sam_image_model != sam_image_model:
        print(f"DEBUG [ComfyUI_Florence2SAM2::lazy_load_models] Old model {loaded_sam_image_model} != new model {sam_image_model} => releasing memory")
        SAM_IMAGE_MODEL.model.cpu()
        del SAM_IMAGE_MODEL
        gc.collect()
        torch.cuda.empty_cache()
        SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE, checkpoint=sam_image_model)
        loaded_sam_image_model = sam_image_model
    if FLORENCE_MODEL is None or FLORENCE_PROCESSOR is None:
        assert FLORENCE_MODEL is None and FLORENCE_PROCESSOR is None
        FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)

    # The models could have been offloaded to RAM by offload_models(); if they're already on `device`, this is a no-op
    SAM_IMAGE_MODEL.model.to(device)
    FLORENCE_MODEL.to(device)
    # FLORENCE_PROCESSOR.to(device) # note a model

def offload_models(delete=False):
    global SAM_IMAGE_MODEL
    global FLORENCE_MODEL
    global FLORENCE_PROCESSOR
    offload_device = mm.unet_offload_device()
    do_gc = False
    if SAM_IMAGE_MODEL is not None:
        if delete:
            SAM_IMAGE_MODEL.model.cpu()
            del SAM_IMAGE_MODEL
            SAM_IMAGE_MODEL = None
            do_gc = True
        else:
            SAM_IMAGE_MODEL.model.to(offload_device)
    if FLORENCE_MODEL is not None:
        if delete:
            FLORENCE_MODEL.cpu()
            del FLORENCE_MODEL
            FLORENCE_MODEL = None
            do_gc = True
        else:
            FLORENCE_MODEL.to(offload_device)
    if FLORENCE_PROCESSOR is not None:
        if delete:
            # FLORENCE_PROCESSOR.cpu()
            del FLORENCE_PROCESSOR
            FLORENCE_PROCESSOR = None
            do_gc = True
        else:
            # FLORENCE_PROCESSOR.cpu()
            pass
    mm.soft_empty_cache()
    if do_gc:
        gc.collect()

def process_image(device: torch.device, sam_image_model: str, image: Image.Image, promt: str, keep_model_loaded: bool) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    lazy_load_models(device, sam_image_model)
    annotated_image, mask_list = _process_image(IMAGE_OPEN_VOCABULARY_DETECTION_MODE, image, promt)
    if mask_list is not None and len(mask_list) > 0:
        mask = np.any(mask_list, axis=0) # Merge masks into a single mask
        mask = (mask * 255).astype(np.uint8)
    else:
        print(f"Florence2SAM2: No objects of class {promt} found in the image.")
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask = Image.fromarray(mask).convert("L") # Convert to 8-bit grayscale
    masked_image = Image.new("RGB", image.size, (0, 0, 0))
    masked_image.paste(image, mask=mask)
    if not keep_model_loaded:
        offload_models()
    return annotated_image, mask, masked_image

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def _process_image(
    mode_dropdown=IMAGE_OPEN_VOCABULARY_DETECTION_MODE, image_input=None, text_input=None
) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
    """
    Process an image with Florence2 and SAM2.

    Note that the models are lazy loaded, so they will not waste time loading during startup, or at all if this method is not called.

    @param mode_dropdown: The mode of the Florence2 model. Must be IMAGE_OPEN_VOCABULARY_DETECTION_MODE.
    @param image_input: The image to process.
    @param text_input: The text prompt to use for the Florence2 model.

    @return: Tuple[Image.Image, Image.Image]: The annotated image, merged mask (Boolean array) of the detected objects
    """
    global SAM_IMAGE_MODEL
    global FLORENCE_MODEL
    global FLORENCE_PROCESSOR

    if not image_input:
        # gr.Info("Please upload an image.")
        return None, None

    if mode_dropdown == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input:
            # gr.Info("Please enter a text prompt.")
            return None, None

        texts = [prompt.strip() for prompt in text_input.split(",")]
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        return annotate_image(image_input, detections), detections.mask

#     if mode_dropdown == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
#         _, result = run_florence_inference(
#             model=FLORENCE_MODEL,
#             processor=FLORENCE_PROCESSOR,
#             device=DEVICE,
#             image=image_input,
#             task=FLORENCE_DETAILED_CAPTION_TASK
#         )
#         caption = result[FLORENCE_DETAILED_CAPTION_TASK]
#         _, result = run_florence_inference(
#             model=FLORENCE_MODEL,
#             processor=FLORENCE_PROCESSOR,
#             device=DEVICE,
#             image=image_input,
#             task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
#             text=caption
#         )
#         detections = sv.Detections.from_lmm(
#             lmm=sv.LMM.FLORENCE_2,
#             result=result,
#             resolution_wh=image_input.size
#         )
#         detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
#         return annotate_image(image_input, detections), caption


# @spaces.GPU(duration=300)
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# def process_video(
#     video_input, text_input, progress=gr.Progress(track_tqdm=True)
# ) -> Optional[str]:
#     if not video_input:
#         gr.Info("Please upload a video.")
#         return None

#     if not text_input:
#         gr.Info("Please enter a text prompt.")
#         return None

#     frame_generator = sv.get_video_frames_generator(video_input)
#     frame = next(frame_generator)
#     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     texts = [prompt.strip() for prompt in text_input.split(",")]
#     detections_list = []
#     for text in texts:
#         _, result = run_florence_inference(
#             model=FLORENCE_MODEL,
#             processor=FLORENCE_PROCESSOR,
#             device=DEVICE,
#             image=frame,
#             task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
#             text=text
#         )
#         detections = sv.Detections.from_lmm(
#             lmm=sv.LMM.FLORENCE_2,
#             result=result,
#             resolution_wh=frame.size
#         )
#         detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)
#         detections_list.append(detections)

#     detections = sv.Detections.merge(detections_list)
#     detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)

#     if len(detections.mask) == 0:
#         gr.Info(
#             "No objects of class {text_input} found in the first frame of the video. "
#             "Trim the video to make the object appear in the first frame or try a "
#             "different text prompt."
#         )
#         return None

#     name = generate_unique_name()
#     frame_directory_path = os.path.join(VIDEO_TARGET_DIRECTORY, name)
#     frames_sink = sv.ImageSink(
#         target_dir_path=frame_directory_path,
#         image_name_pattern="{:05d}.jpeg"
#     )

#     video_info = sv.VideoInfo.from_video_path(video_input)
#     video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
#     video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

#     frames_generator = sv.get_video_frames_generator(video_input)
#     with frames_sink:
#         for frame in tqdm(
#                 frames_generator,
#                 total=video_info.total_frames,
#                 desc="splitting video into frames"
#         ):
#             frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
#             frames_sink.save_image(frame)

#     inference_state = SAM_VIDEO_MODEL.init_state(
#         video_path=frame_directory_path,
#         device=DEVICE
#     )

#     for mask_index, mask in enumerate(detections.mask):
#         _, object_ids, mask_logits = SAM_VIDEO_MODEL.add_new_mask(
#             inference_state=inference_state,
#             frame_idx=0,
#             obj_id=mask_index,
#             mask=mask
#         )

#     video_path = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")
#     frames_generator = sv.get_video_frames_generator(video_input)
#     masks_generator = SAM_VIDEO_MODEL.propagate_in_video(inference_state)
#     with sv.VideoSink(video_path, video_info=video_info) as sink:
#         for frame, (_, tracker_ids, mask_logits) in zip(frames_generator, masks_generator):
#             frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
#             masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
#             if len(masks.shape) == 4:
#                 masks = np.squeeze(masks, axis=1)

#             detections = sv.Detections(
#                 xyxy=sv.mask_to_xyxy(masks=masks),
#                 mask=masks,
#                 class_id=np.array(tracker_ids)
#             )
#             annotated_frame = frame.copy()
#             annotated_frame = MASK_ANNOTATOR.annotate(
#                 scene=annotated_frame, detections=detections)
#             annotated_frame = BOX_ANNOTATOR.annotate(
#                 scene=annotated_frame, detections=detections)
#             sink.write_frame(annotated_frame)

#     delete_directory(frame_directory_path)
#     return video_path


# with gr.Blocks() as demo:
#     gr.Markdown(MARKDOWN)
#     with gr.Tab("Image"):
#         image_processing_mode_dropdown_component = gr.Dropdown(
#             choices=IMAGE_INFERENCE_MODES,
#             value=IMAGE_INFERENCE_MODES[0],
#             label="Mode",
#             info="Select a mode to use.",
#             interactive=True
#         )
#         with gr.Row():
#             with gr.Column():
#                 image_processing_image_input_component = gr.Image(
#                     type='pil', label='Upload image')
#                 image_processing_text_input_component = gr.Textbox(
#                     label='Text prompt',
#                     placeholder='Enter comma separated text prompts')
#                 image_processing_submit_button_component = gr.Button(
#                     value='Submit', variant='primary')
#             with gr.Column():
#                 image_processing_image_output_component = gr.Image(
#                     type='pil', label='Image output')
#                 image_processing_text_output_component = gr.Textbox(
#                     label='Caption output', visible=False)

#         with gr.Row():
#             gr.Examples(
#                 fn=process_image,
#                 examples=IMAGE_PROCESSING_EXAMPLES,
#                 inputs=[
#                     image_processing_mode_dropdown_component,
#                     image_processing_image_input_component,
#                     image_processing_text_input_component
#                 ],
#                 outputs=[
#                     image_processing_image_output_component,
#                     image_processing_text_output_component
#                 ],
#                 run_on_click=True
#             )
#     with gr.Tab("Video"):
#         video_processing_mode_dropdown_component = gr.Dropdown(
#             choices=VIDEO_INFERENCE_MODES,
#             value=VIDEO_INFERENCE_MODES[0],
#             label="Mode",
#             info="Select a mode to use.",
#             interactive=True
#         )
#         with gr.Row():
#             with gr.Column():
#                 video_processing_video_input_component = gr.Video(
#                     label='Upload video')
#                 video_processing_text_input_component = gr.Textbox(
#                     label='Text prompt',
#                     placeholder='Enter comma separated text prompts')
#                 video_processing_submit_button_component = gr.Button(
#                     value='Submit', variant='primary')
#             with gr.Column():
#                 video_processing_video_output_component = gr.Video(
#                     label='Video output')
#         with gr.Row():
#             gr.Examples(
#                 fn=process_video,
#                 examples=VIDEO_PROCESSING_EXAMPLES,
#                 inputs=[
#                     video_processing_video_input_component,
#                     video_processing_text_input_component
#                 ],
#                 outputs=video_processing_video_output_component,
#                 run_on_click=True
#             )

#     image_processing_submit_button_component.click(
#         fn=process_image,
#         inputs=[
#             image_processing_mode_dropdown_component,
#             image_processing_image_input_component,
#             image_processing_text_input_component
#         ],
#         outputs=[
#             image_processing_image_output_component,
#             image_processing_text_output_component
#         ]
#     )
#     image_processing_text_input_component.submit(
#         fn=process_image,
#         inputs=[
#             image_processing_mode_dropdown_component,
#             image_processing_image_input_component,
#             image_processing_text_input_component
#         ],
#         outputs=[
#             image_processing_image_output_component,
#             image_processing_text_output_component
#         ]
#     )
#     image_processing_mode_dropdown_component.change(
#         on_mode_dropdown_change,
#         inputs=[image_processing_mode_dropdown_component],
#         outputs=[
#             image_processing_text_input_component,
#             image_processing_text_output_component
#         ]
#     )
#     video_processing_submit_button_component.click(
#         fn=process_video,
#         inputs=[
#             video_processing_video_input_component,
#             video_processing_text_input_component
#         ],
#         outputs=video_processing_video_output_component
#     )
#     video_processing_text_input_component.submit(
#         fn=process_video,
#         inputs=[
#             video_processing_video_input_component,
#             video_processing_text_input_component
#         ],
#         outputs=video_processing_video_output_component
#     )

# demo.launch(debug=False, show_error=True)
