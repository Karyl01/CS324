#!/usr/bin/env python3
"""
数据集处理工具
"""

import os
import random
import glob
from typing import List, Tuple, Optional
from pathlib import Path

def split_dataset(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    分割数据集为训练、验证和测试集

    Args:
        data_dir: 数据目录路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_files, val_files, test_files) 文件名列表
    """
    # 检查比例总和
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练、验证和测试集比例之和必须为1.0")

    # 设置随机种子
    random.seed(seed)

    # 获取所有图像文件
    images_dir = os.path.join(data_dir, "images")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    # 随机打乱
    random.shuffle(image_files)

    # 计算分割点
    total_files = len(image_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # 分割数据集
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    print(f"数据集分割完成:")
    print(f"  总文件数: {total_files}")
    print(f"  训练集: {len(train_files)} ({len(train_files)/total_files:.1%})")
    print(f"  验证集: {len(val_files)} ({len(val_files)/total_files:.1%})")
    print(f"  测试集: {len(test_files)} ({len(test_files)/total_files:.1%})")

    return train_files, val_files, test_files

def create_dataset_splits(data_dir: str, output_dir: str = None):
    """
    创建数据集分割文件

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = data_dir

    # 分割数据集
    train_files, val_files, test_files = split_dataset(data_dir)

    # 创建训练文件列表
    train_txt = os.path.join(output_dir, "train.txt")
    val_txt = os.path.join(output_dir, "val.txt")
    test_txt = os.path.join(output_dir, "test.txt")

    # 写入文件列表
    def write_file_list(file_path: str, file_list: List[str]):
        with open(file_path, 'w') as f:
            for file_path_abs in file_list:
                # 相对路径
                rel_path = os.path.relpath(file_path_abs, data_dir)
                f.write(f"./{rel_path}\n")

    write_file_list(train_txt, train_files)
    write_file_list(val_txt, val_files)
    write_file_list(test_txt, test_files)

    print(f"文件列表已保存到:")
    print(f"  {train_txt}")
    print(f"  {val_txt}")
    print(f"  {test_txt}")

def validate_dataset(data_dir: str) -> bool:
    """
    验证数据集完整性

    Args:
        data_dir: 数据目录

    Returns:
        数据集是否有效
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return False

    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return False

    # 获取图像和标签文件
    image_files = set()
    label_files = set()

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.update([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(images_dir, ext))])

    label_files.update([os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(labels_dir, "*.txt"))])

    # 检查匹配
    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        print(f"警告: {len(missing_labels)} 个图像文件缺少对应的标签文件")
        for missing in list(missing_labels)[:5]:  # 只显示前5个
            print(f"  {missing}")

    if missing_images:
        print(f"警告: {len(missing_images)} 个标签文件缺少对应的图像文件")
        for missing in list(missing_images)[:5]:  # 只显示前5个
            print(f"  {missing}")

    valid_files = image_files & label_files
    print(f"数据集验证完成:")
    print(f"  有效样本数: {len(valid_files)}")
    print(f"  缺少标签的图像: {len(missing_labels)}")
    print(f"  缺少图像的标签: {len(missing_images)}")

    return len(valid_files) > 0

def analyze_dataset(data_dir: str):
    """
    分析数据集统计信息

    Args:
        data_dir: 数据目录
    """
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return

    class_counts = {}
    total_objects = 0
    total_images = 0

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for label_file in label_files:
        total_images += 1

        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_objects += 1

        except Exception as e:
            print(f"警告: 无法读取标签文件 {label_file}: {e}")

    print("数据集统计信息:")
    print(f"  总图像数: {total_images}")
    print(f"  总目标数: {total_objects}")
    print(f"  平均每张图像目标数: {total_objects/total_images:.2f}")
    print("\n各类别统计:")

    # 从配置中获取类别名称
    try:
        from utils.config import CLASS_NAMES
        for class_id, count in sorted(class_counts.items()):
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            percentage = (count / total_objects) * 100
            print(f"  {class_name:15}: {count:4d} ({percentage:5.1f}%)")
    except ImportError:
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100
            print(f"  Class {class_id}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    # 测试数据集工具
    data_dir = "/mnt/d/Study/CS324/yolo10/data"

    print("验证数据集...")
    is_valid = validate_dataset(data_dir)

    if is_valid:
        print("\n分析数据集...")
        analyze_dataset(data_dir)

        print("\n创建数据集分割...")
        create_dataset_splits(data_dir)