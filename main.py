import warnings
from glob import glob

import gradio as gr

from src.inference.inference_yolo_ncnn import NCNN_MODEL
from src.utils import clear_output_directory

warnings.filterwarnings('ignore')

SAMPLE_VIDEO_NAME = 'test_2sec'
SAMPLE_IMAGE_NAME = 'sample_2'

def process_video(video):
    clear_output_directory()
    NCNN_MODEL.simple_predict(video, save=True)
    output_files = glob('runs/detect/predict/*.avi')
    if output_files:
        return output_files[0]
    else:
        return None

def process_image(image):
    clear_output_directory()
    NCNN_MODEL.simple_predict(image, save=True)
    output_files = glob('runs/detect/predict/*.jpg')
    if output_files:
        return output_files[0]
    else:
        return None


with gr.Blocks() as demo:
    gr.Markdown("# Detection of Road Signs")
    gr.Markdown("To start the process, upload a video (mp4) or an image (jpg) file.")
    gr.Markdown("### Note: Run this on a local device.")

    video_interface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(),
        outputs="playable_video",
        examples=[f"samples/video/{SAMPLE_VIDEO_NAME}.mov"],
        cache_examples=True,
        live=True,
        theme="huggingface",
        title="Video Processing"
    )

    image_interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="pil"),
        outputs="image",
        examples=[f"samples/images/{SAMPLE_IMAGE_NAME}.jpg"],
        theme="huggingface",
        title="Image Processing"
    )

if __name__ == "__main__":
    demo.launch(share=True)
