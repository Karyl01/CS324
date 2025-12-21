#!/usr/bin/env python3
"""
YOLOv10n手势识别项目配置文件
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径配置
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, "labels")
DATASET_CONFIG = os.path.join(DATA_ROOT, "dataset.yaml")

# 模型配置
MODEL_SIZE = "n"  # YOLOv10n (nano版本)
NUM_CLASSES = 8
IMG_SIZE = 416
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# 训练配置
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
# 设备检测（延迟导入以避免依赖问题）
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "detect")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 确保输出目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 手势类别名称
CLASS_NAMES = [
    "up",
    "down",
    "left",
    "right",
    "front",
    "back",
    "clockwise",
    "anticlockwise"
]

# 类别颜色映射（用于可视化）
CLASS_COLORS = [
    (255, 0, 0),    # 红色 - up
    (0, 255, 0),    # 绿色 - down
    (0, 0, 255),    # 蓝色 - left
    (255, 255, 0),  # 黄色 - right
    (255, 0, 255),  # 紫色 - front
    (0, 255, 255),  # 青色 - back
    (128, 0, 128),  # 深紫色 - clockwise
    (255, 165, 0)   # 橙色 - anticlockwise
]

# 模型下载配置
YOLOv10_MODEL_URL = {
    "n": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "s": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "m": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
    "b": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
    "l": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
    "x": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt"
}