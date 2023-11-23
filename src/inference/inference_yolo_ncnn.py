import sys
import logging as log
from pathlib import Path

import cv2
import numpy as np

from .models.yolo_ncnn import YoloNCNN

sys.path.append(str(Path(__file__).resolve().parents[2].joinpath('configs')))
from models_configs.model_configurator import ModelConfig
from logger_conf import configure_logger


configure_logger()
ROOT_DIR = Path(__file__).resolve().parents[2]

log.info('[NCNN] Setting up models config')
MODEL_CONF = ModelConfig()

MODEL_WEIGHTS = ROOT_DIR / MODEL_CONF.properties['yolov8n']['ncnn']['model_dir']
log.info(f'[NCNN] Loading model: {MODEL_WEIGHTS}')
NCNN_MODEL = YoloNCNN(MODEL_WEIGHTS)


def detect_get_bounding_boxes(input, raw_output=False):
    """
    Performs NCNN inference on input. Can return raw boxes coodrinates
    (use raw_output flag) or rectangles as bounding boxes.
    Params:
        input: image. np.ndarray
    Returns:
        input_image: The combined image. np.ndarray
    """

    log.info('Inference: detection')
    results = NCNN_MODEL.simple_predict(input)
    out_boxes = []
    rects = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        out_boxes += boxes
        for box in boxes:
            name = result.names[int(box.cls[0])]
            rects.append([box.xyxy[0].astype(int), name])

    log.info(F'Detected {len(rects)} objects')
    if raw_output:
        return out_boxes

    return rects

def detect_draw_boxes(input_image):
    """
    Combines image and its detected bounding boxes into a single image.
    Params:
        input_image: image. np.ndarray
    Returns:
        input_image: The combined image. np.ndarray
    """
    log.info('Drawing bounding boxes')
    rects = detect_get_bounding_boxes(input_image)
    for r in rects:
        cv2.rectangle(input_image, r[0][:2], r[0][2:], (255, 255, 255), 2)
        cv2.putText(input_image, r[1], (r[0][0], r[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return input_image
