#!/usr/bin/env python3
"""
YOLOv10n手势识别项目设置测试脚本
"""

import os
import sys
import importlib
from pathlib import Path

def test_imports():
    """测试必要模块的导入"""
    print("测试模块导入...")

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 导入失败")
        return False

    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV 导入失败")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy 导入失败")
        return False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO 导入成功")
    except ImportError:
        print("✗ Ultralytics YOLO 导入失败")
        return False

    return True

def test_project_structure():
    """测试项目结构"""
    print("\n测试项目结构...")

    required_dirs = [
        "data",
        "data/images",
        "data/labels",
        "utils",
        "models"
    ]

    required_files = [
        "train_yolov10n.py",
        "predict_yolov10n.py",
        "requirements.txt",
        "README.md",
        "data/dataset.yaml",
        "utils/config.py",
        "models/yolov10n_custom.py"
    ]

    # 检查目录
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (缺失)")
            return False

    # 检查文件
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            return False

    return True

def test_data_files():
    """测试数据文件"""
    print("\n测试数据文件...")

    images_dir = "data/images"
    labels_dir = "data/labels"

    # 统计文件数量
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]

    print(f"✓ 图像文件数量: {len(image_files)}")
    print(f"✓ 标签文件数量: {len(label_files)}")

    if len(image_files) == 0:
        print("✗ 没有找到图像文件")
        return False

    if len(label_files) == 0:
        print("✗ 没有找到标签文件")
        return False

    # 检查第一个标签文件格式
    first_label = os.path.join(labels_dir, label_files[0])
    try:
        with open(first_label, 'r') as f:
            content = f.read().strip()
            if content:
                parts = content.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    print(f"✓ 标签文件格式正确 (class_id={class_id}, coords={coords})")
                else:
                    print("✗ 标签文件格式不正确")
                    return False
            else:
                print(f"✗ 标签文件为空: {first_label}")
                return False
    except Exception as e:
        print(f"✗ 无法读取标签文件 {first_label}: {e}")
        return False

    return True

def test_config():
    """测试配置文件"""
    print("\n测试配置文件...")

    try:
        from utils.config import (
            NUM_CLASSES, IMG_SIZE, CLASS_NAMES,
            CLASS_COLORS, DATASET_CONFIG
        )

        print(f"✓ NUM_CLASSES: {NUM_CLASSES}")
        print(f"✓ IMG_SIZE: {IMG_SIZE}")
        print(f"✓ CLASS_NAMES: {len(CLASS_NAMES)} 类")
        print(f"✓ CLASS_COLORS: {len(CLASS_COLORS)} 种颜色")
        print(f"✓ DATASET_CONFIG: {DATASET_CONFIG}")

        return True

    except ImportError as e:
        print(f"✗ 配置文件导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")

    try:
        from models.yolov10n_custom import create_model

        # 创建模型（不使用预训练权重，避免下载）
        print("正在创建YOLOv10n模型...")
        detector = create_model("n", pretrained=False)

        model_info = detector.get_model_info()
        print(f"✓ 模型创建成功")
        print(f"  模型类型: {model_info['model_type']}")
        print(f"  类别数量: {model_info['num_classes']}")
        print(f"  图像尺寸: {model_info['image_size']}")

        return True

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_gpu_availability():
    """测试GPU可用性"""
    print("\n测试GPU可用性...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            print(f"✓ GPU 可用: {gpu_count} 个设备")
            print(f"✓ 当前设备: {device_name}")

            # 测试GPU内存
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3

            print(f"  已分配内存: {memory_allocated:.2f} GB")
            print(f"  缓存内存: {memory_cached:.2f} GB")

        else:
            print("⚠ GPU 不可用，将使用CPU")

        return True

    except Exception as e:
        print(f"✗ GPU检测失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("YOLOv10n 手势识别项目设置测试")
    print("=" * 60)

    tests = [
        ("模块导入", test_imports),
        ("项目结构", test_project_structure),
        ("数据文件", test_data_files),
        ("配置文件", test_config),
        ("模型创建", test_model_creation),
        ("GPU可用性", test_gpu_availability)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试出错: {e}")

    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！项目设置正确，可以开始训练。")
        return True
    else:
        print("❌ 部分测试失败，请检查项目设置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)