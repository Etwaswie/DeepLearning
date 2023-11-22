# Глубокое обучение на практике

* Глубоко обучаем
* На практике

## Сравнение инференса
Устройство: Raspberry Pi 4B (8 GB RAM, 1.5 GHz CPU)
| **Framework**       | **Model**    | **Task Type** | **FPS** | **Average inference time, ms** |
|---------------------|--------------|---------------|---------|--------------------------------|
| NCNN                | YOLO v8 nano | detection     | 2.13    | 469                            |
| NCNN                | YOLO v8 nano | segmentation  | 1.66    | 603                            |
| PyTorch             | YOLO v8 nano | detection     | 1.12    | 892                            |
| PyTorch             | YOLO v8 nano | segmentation  | 1.09    | 913                            |
| PyTorch+Torchscript | YOLO v8 nano | detection     | -       | 1322                           |
| ONNX Runtime        | YOLO v8 nano | detection     | -       | 526                            |
| OpenVINO            | YOLO v8 nano | detection     | -       | 1036                           |

В MVP используется NCNN как демонстрирующий наибольшую производительность на ARM.
