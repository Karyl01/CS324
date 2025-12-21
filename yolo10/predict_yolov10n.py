#!/usr/bin/env python3
"""
YOLOv10n手势识别摄像头推理脚本
实时摄像头手势检测和可视化
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from models.yolov10n_custom import create_model
from utils.config import (
    NUM_CLASSES, IMG_SIZE, CONF_THRESHOLD,
    IOU_THRESHOLD, CLASS_NAMES, CLASS_COLORS
)

class GestureDetector:
    """手势检测器类"""

    def __init__(self, model_path, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
        """
        初始化手势检测器

        Args:
            model_path: 训练好的模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES
        self.class_colors = CLASS_COLORS

        # 加载模型
        self.model = self._load_model()

        # FPS计算
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            # 使用YOLO加载模型
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            print(f"成功加载模型: {self.model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def _calculate_fps(self):
        """计算FPS"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time

        if elapsed_time >= 1.0:  # 每秒更新一次FPS
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time

        return self.current_fps

    def _draw_detections(self, frame, results):
        """在图像上绘制检测结果"""
        annotated_frame = frame.copy()

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # 获取置信度和类别
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 跳过低置信度检测
                    if conf < self.conf_threshold:
                        continue

                    # 获取类别名称和颜色
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    color = self.class_colors[class_id] if class_id < len(self.class_colors) else (0, 255, 0)

                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签背景
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color,
                        -1
                    )

                    # 绘制标签文字
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        return annotated_frame

    def _draw_info(self, frame):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        info_height = 60

        # 创建信息面板背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, info_height), (0, 0, 0), -1)
        alpha = 0.7
        frame[0:info_height, :] = cv2.addWeighted(frame[0:info_height, :], alpha, overlay[0:info_height, :], 1 - alpha, 0)

        # 显示FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示模型信息
        model_text = f"Model: {os.path.basename(self.model_path)}"
        cv2.putText(frame, model_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示操作提示
        help_text = "Press 'q' to quit, 's' to save frame"
        cv2.putText(frame, help_text, (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_frame(self, frame):
        """处理单帧图像"""
        # 模型推理
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=IMG_SIZE,
            verbose=False
        )

        # 绘制检测结果
        annotated_frame = self._draw_detections(frame, results)

        # 绘制信息面板
        annotated_frame = self._draw_info(annotated_frame)

        # 计算FPS
        self._calculate_fps()

        return annotated_frame

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv10n手势识别摄像头推理")

    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD,
                        help='IOU阈值')

    # 摄像头参数
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头设备ID')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='摄像头分辨率 (格式: WxH)')

    # 显示参数
    parser.add_argument('--fullscreen', action='store_true',
                        help='全屏显示')
    parser.add_argument('--save-dir', type=str, default='captures',
                        help='截图保存目录')

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("YOLOv10n 手势识别摄像头推理")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"置信度阈值: {args.conf}")
    print(f"IOU阈值: {args.iou}")
    print(f"摄像头ID: {args.camera}")
    print(f"分辨率: {args.resolution}")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)

    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print("错误: 分辨率格式不正确，应为 WxH")
        sys.exit(1)

    # 创建截图保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化手势检测器
    try:
        detector = GestureDetector(args.model, args.conf, args.iou)
    except Exception as e:
        print(f"检测器初始化失败: {e}")
        sys.exit(1)

    # 初始化摄像头
    print("初始化摄像头...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {args.camera}")
        sys.exit(1)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 等待摄像头稳定
    time.sleep(2)

    print("开始实时检测 (按 'q' 退出, 's' 保存截图)...")

    frame_count = 0
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法从摄像头读取帧")
                break

            # 处理帧
            processed_frame = detector.process_frame(frame)

            # 显示结果
            if args.fullscreen:
                cv2.namedWindow('YOLOv10n Gesture Detection', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('YOLOv10n Gesture Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow('YOLOv10n Gesture Detection', processed_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存截图
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(args.save_dir, f"gesture_capture_{timestamp}.jpg")
                cv2.imwrite(save_path, processed_frame)
                print(f"截图已保存: {save_path}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n检测被用户中断")

    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main()