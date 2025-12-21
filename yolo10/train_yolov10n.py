#!/usr/bin/env python3
"""
YOLOv10n手势识别训练脚本
使用转换后的YOLO格式数据集进行模型训练
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from models.yolov10n_custom import create_model
from utils.config import (
    NUM_CLASSES, IMG_SIZE, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, DEVICE, DATASET_CONFIG, OUTPUT_DIR
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv10n手势识别训练")

    # 数据参数
    parser.add_argument('--data', type=str, default=DATASET_CONFIG,
                        help='数据集配置文件路径')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE,
                        help='输入图像尺寸')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='学习率')

    # 模型参数
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'b', 'l', 'x'],
                        help='YOLOv10模型大小')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重')

    # 设备和输出
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='训练设备 (cpu, cuda, auto)')
    parser.add_argument('--name', type=str, default='yolov10n_gesture',
                        help='实验名称')
    parser.add_argument('--project', type=str, default=OUTPUT_DIR,
                        help='项目输出目录')

    # 训练策略
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='优化器类型')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--save-period', type=int, default=10,
                        help='模型保存间隔')

    # 数据增强
    parser.add_argument('--augment', action='store_true',
                        help='启用数据增强')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic数据增强概率')

    return parser.parse_args()

def main():
    """主训练函数"""
    args = parse_args()

    print("=" * 60)
    print("YOLOv10n 手势识别训练开始")
    print("=" * 60)
    print(f"训练配置:")
    print(f"  数据集配置: {args.data}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批量大小: {args.batch}")
    print(f"  学习率: {args.lr}")
    print(f"  模型大小: YOLOv10{args.model}")
    print(f"  优化器: {args.optimizer}")
    print(f"  设备: {args.device}")
    print("=" * 60)

    # 检查数据集配置文件
    if not os.path.exists(args.data):
        print(f"错误: 数据集配置文件不存在: {args.data}")
        sys.exit(1)

    # 检查数据目录
    data_dir = os.path.dirname(args.data)
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"错误: 数据目录不存在")
        print(f"  图像目录: {images_dir}")
        print(f"  标签目录: {labels_dir}")
        sys.exit(1)

    # 统计数据集
    num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_labels = len([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])

    print(f"数据集统计:")
    print(f"  图像数量: {num_images}")
    print(f"  标签数量: {num_labels}")
    print(f"  类别数量: {NUM_CLASSES}")
    print("-" * 60)

    # 创建模型
    print("创建YOLOv10模型...")
    try:
        detector = create_model(args.model, pretrained=args.pretrained)
        print(f"成功创建YOLOv10{args.model}模型")
    except Exception as e:
        print(f"模型创建失败: {e}")
        sys.exit(1)

    # 设置训练参数
    training_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'name': args.name,
        'project': args.project,
        'optimizer': args.optimizer,
        'patience': args.patience,
        'save_period': args.save_period,
        'lr0': args.lr,
        'plots': True,
        'verbose': True,
        'exist_ok': True
    }

    # 数据增强设置
    if args.augment:
        training_args.update({
            'hsv_h': 0.015,  # HSV色调增强
            'hsv_s': 0.7,    # HSV饱和度增强
            'hsv_v': 0.4,    # HSV明度增强
            'degrees': 0.0,  # 旋转角度
            'translate': 0.1,  # 平移
            'scale': 0.5,    # 缩放
            'shear': 0.0,    # 剪切
            'perspective': 0.0,  # 透视
            'flipud': 0.0,   # 垂直翻转
            'fliplr': 0.5,   # 水平翻转
            'mosaic': args.mosaic,  # Mosaic增强
            'mixup': 0.0,    # Mixup增强
        })

    # 开始训练
    print("开始训练...")
    start_time = time.time()

    try:
        results = detector.model.train(**training_args)

        end_time = time.time()
        training_time = end_time - start_time

        print("=" * 60)
        print("训练完成!")
        print(f"训练时长: {training_time:.2f} 秒 ({training_time/3600:.2f} 小时)")
        print(f"最佳模型保存在: {results.save_dir}")

        # 打印训练结果摘要
        if hasattr(results, 'results_dict'):
            print("\n训练结果摘要:")
            results_dict = results.results_dict
            for metric, value in results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()