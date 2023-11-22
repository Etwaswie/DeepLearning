import sys
import logging as log
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('inference')))
from inference_yolo_ncnn import detect_get_bounding_boxes
sys.path.append(str(Path(__file__).resolve().parents[2].joinpath('configs')))
from models_configs.model_configurator import ModelConfig
from logger_conf import configure_logger

configure_logger()

NUM_ITERATIONS = 10


def benchmark_model(model, model_name, num_iterations, test_data):
    total_time = 0

    for _ in range(num_iterations):
        start_time = time.time()
        res = model(test_data)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_time += inference_time

    fps = num_iterations / (total_time / 1000)  # Convert total time to seconds
    average_time = total_time / num_iterations

    log.info(f'{model_name} Detection FPS: {fps:.2f}')
    log.info(f'{model_name} Detection Average Inference Time: {average_time:.4f} seconds')
    return model_name, f"{fps:.2f}", f"{average_time:.4f}"


def main():
    log.info('Starting benchmark: YOLOv8n detection+classification')
    root_dir = Path(__file__).resolve().parents[2]
    model_conf = ModelConfig()
    model_weights_pytorch = root_dir / model_conf.properties['yolov8n']['pytorch']['weights']
    model_weights_ort = root_dir / model_conf.properties['yolov8n']['onnx']['weights']

    log.info(f'[PyTorch] Loading model: {model_weights_pytorch}')
    pytorch_model_det = YOLO(model_weights_pytorch)
    log.info(f'[ORT] Loading model: {model_weights_pytorch}')
    ort_model_det = YOLO(model_weights_ort)

    test_data = np.random.rand(640, 640, 3) * 255  # Generate random image

    pytorch_res = benchmark_model(pytorch_model_det, 'PyTorch', NUM_ITERATIONS, test_data)
    ncnn_res = benchmark_model(detect_get_bounding_boxes, 'NCNN', NUM_ITERATIONS, test_data)
    ort_res = benchmark_model(ort_model_det, 'ORT', NUM_ITERATIONS, test_data)

    log.info('==============Benchmark Results=================')
    log.info('| Framework  | Model   | Image Size | Inference Time (ms) | FPS   |')
    log.info('|------------|---------|------------|---------------------|-------|')

    log.info(f'|  PyTorch   | YOLOv8n | 640x640    | {pytorch_res[2]}          | {pytorch_res[1]}')
    log.info(f'|    NCNN    | YOLOv8n | 640x640    | {ncnn_res[2]}           | {ncnn_res[1]}')
    log.info(f'|     ORT    | YOLOv8n | 640x640    | {ort_res[2]}           | {ort_res[1]}')

    log.info('===============================================')

if __name__ == "__main__":
    main()
