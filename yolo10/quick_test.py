#!/usr/bin/env python3
"""
简化的项目测试脚本
"""

import os

def test_basic_setup():
    """基本设置测试"""
    print("YOLOv10n 手势识别项目快速测试")
    print("=" * 50)

    # 检查项目结构
    required_items = [
        "data/images", "data/labels", "data/dataset.yaml",
        "train_yolov10n.py", "predict_yolov10n.py",
        "utils/config.py", "models/yolov10n_custom.py"
    ]

    print("检查项目结构:")
    for item in required_items:
        if os.path.exists(item):
            print(f"  ✓ {item}")
        else:
            print(f"  ✗ {item} (缺失)")

    # 检查数据集
    print("\n数据集统计:")
    if os.path.exists("data/images"):
        images = [f for f in os.listdir("data/images")
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  图像文件: {len(images)} 个")

    if os.path.exists("data/labels"):
        labels = [f for f in os.listdir("data/labels") if f.lower().endswith('.txt')]
        print(f"  标签文件: {len(labels)} 个")

    # 检查配置
    print("\n配置信息:")
    try:
        from utils.config import NUM_CLASSES, IMG_SIZE, CLASS_NAMES
        print(f"  类别数量: {NUM_CLASSES}")
        print(f"  图像尺寸: {IMG_SIZE}")
        print(f"  类别名称: {', '.join(CLASS_NAMES)}")
    except Exception as e:
        print(f"  配置加载失败: {e}")

    print("\n✓ 基本项目结构检查完成")
    print("\n使用方法:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 训练模型: python train_yolov10n.py --model n --epochs 50 --batch 16")
    print("3. 推理测试: python predict_yolov10n.py --model <model_path> --camera 0")

if __name__ == "__main__":
    test_basic_setup()