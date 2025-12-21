#!/usr/bin/env python3
"""
YOLOv10n自定义模型实现
基于Ultralytics YOLOv10框架，针对手势识别任务进行适配
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from utils.config import NUM_CLASSES, IMG_SIZE

class YOLOv10nGestureDetector:
    """
    YOLOv10n手势检测器
    """

    def __init__(self, model_path="yolov10n.pt", num_classes=NUM_CLASSES):
        """
        初始化YOLOv10n手势检测器

        Args:
            model_path: 预训练模型路径
            num_classes: 类别数量
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.img_size = IMG_SIZE

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载并配置模型"""
        # 加载预训练的YOLOv10n模型
        model = YOLO(self.model_path)

        # 修改模型头以适应我们的类别数量
        if hasattr(model.model, 'model') and hasattr(model.model.model[-1], 'nc'):
            model.model.model[-1].nc = self.num_classes

        return model

    def train(self, data_yaml, epochs=100, imgsz=416, batch=32, device='auto'):
        """
        训练模型

        Args:
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批量大小
            device: 设备类型
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name="yolov10n_gesture",
            save_period=10,  # 每10轮保存一次
            plots=True,      # 生成训练图表
            verbose=True
        )
        return results

    def predict(self, source, conf=0.5, iou=0.45, save=False):
        """
        进行预测

        Args:
            source: 输入源（图像路径、视频路径、摄像头ID等）
            conf: 置信度阈值
            iou: IOU阈值
            save: 是否保存结果
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            save=save,
            verbose=False
        )
        return results

    def export(self, format="onnx"):
        """
        导出模型

        Args:
            format: 导出格式 (onnx, torchscript, etc.)
        """
        return self.model.export(format=format)

    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_type": "YOLOv10n",
            "num_classes": self.num_classes,
            "image_size": self.img_size,
            "parameters": sum(p.numel() for p in self.model.model.parameters()),
            "model_path": self.model_path
        }

def create_model(model_size="n", pretrained=True):
    """
    创建YOLOv10模型

    Args:
        model_size: 模型大小 (n, s, m, b, l, x)
        pretrained: 是否使用预训练权重
    """
    if pretrained:
        model_name = f"yolov10{model_size}.pt"
        detector = YOLOv10nGestureDetector(model_name)
    else:
        # 创建未训练的模型结构
        model_name = f"yolov10{model_size}.pt"
        detector = YOLOv10nGestureDetector(model_name)

    return detector

if __name__ == "__main__":
    # 测试模型创建
    detector = create_model("n", pretrained=True)
    model_info = detector.get_model_info()

    print("模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")