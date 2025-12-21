#!/usr/bin/env python3
"""
将VOC格式的标注数据转换为YOLO格式
"""

import os
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
import shutil

# 手势类别映射
CLASS_MAPPING = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "front": 4,
    "back": 5,
    "clockwise": 6,
    "anticlockwise": 7
}

def convert_voc_to_yolo(xml_file, output_dir):
    """
    将单个VOC XML文件转换为YOLO格式

    Args:
        xml_file: XML标注文件路径
        output_dir: 输出目录
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # 获取图像文件名
    filename = root.find('filename').text
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    output_path = os.path.join(output_dir, txt_filename)

    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[class_name]

            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转换为YOLO格式 (归一化的中心坐标和宽高)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 写入YOLO格式
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def copy_images(src_dir, dst_dir):
    """复制图像文件"""
    os.makedirs(dst_dir, exist_ok=True)

    image_files = glob(os.path.join(src_dir, "*.jpg"))
    image_files.extend(glob(os.path.join(src_dir, "*.jpeg")))
    image_files.extend(glob(os.path.join(src_dir, "*.png")))

    for img_file in tqdm(image_files, desc="复制图像文件"):
        shutil.copy2(img_file, dst_dir)

def main():
    # 路径配置
    voc_path = "/mnt/d/Study/CS324/YoloGesture-1.1/VOC2007"
    xml_dir = os.path.join(voc_path, "Annotations")
    img_dir = os.path.join(voc_path, "JPEGImages")

    # 输出路径
    project_root = "/mnt/d/Study/CS324/yolo10"
    output_img_dir = os.path.join(project_root, "data/images")
    output_label_dir = os.path.join(project_root, "data/labels")

    # 创建输出目录
    os.makedirs(output_label_dir, exist_ok=True)

    print("开始转换VOC标注到YOLO格式...")

    # 转换所有XML文件
    xml_files = glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in tqdm(xml_files, desc="转换标注文件"):
        convert_voc_to_yolo(xml_file, output_label_dir)

    print(f"转换完成！生成了 {len(xml_files)} 个标注文件")

    print("复制图像文件...")
    copy_images(img_dir, output_img_dir)

    print("数据准备完成！")
    print(f"图像文件: {output_img_dir}")
    print(f"标注文件: {output_label_dir}")

if __name__ == "__main__":
    main()