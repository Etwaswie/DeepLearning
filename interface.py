import gradio as gr
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

model = YOLO("best.pt") # write your path to model

def video_inferance(video):
    model.predict(video, save=True)
    # '/ write your absolute path to predict file /runs/detect/predict/test.mp4'
    return '/Users/alsukurmakaeva/Desktop/ITMO/cv/runs/detect/predict/test.mp4'   

with gr.Blocks() as demo:
    gr.Markdown("# Детекция дорожных знаков")
    gr.Markdown(" Для запуска работы загрузите видео файл, предварительно его переимновав в формате:")
    gr.Markdown(" test.mp4 или test.mov")
    gr.Markdown("### Ограничения! - запускать необходимо на локальном устройстве, google colab не подойдет")
    resut = gr.Interface(video_inferance,
                    gr.Video(),
                    "playable_video",
                    examples=[
                        "video/test.mov"], #create this dir and put on file from (https://drive.google.com/file/d/1WQjDMEuj9d4fKHV5F5XVTuswsDclzBDp/view?usp=sharing)
                    cache_examples=True)

if __name__ == "__main__":
    demo.launch(share=True)