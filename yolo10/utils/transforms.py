#!/usr/bin/env python3
"""
数据增强和变换工具
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional

class GestureTransforms:
    """手势识别专用数据增强"""

    def __init__(self, img_size: int = 416):
        self.img_size = img_size

    def resize_and_pad(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        调整图像大小并保持长宽比

        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)

        Returns:
            调整大小后的图像
        """
        if target_size is None:
            target_size = (self.img_size, self.img_size)

        h, w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 调整大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充图像
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # 灰色填充

        # 计算填充位置
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # 放置调整后的图像
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    def random_rotation(self, image: np.ndarray, angle_range: Tuple[float, float] = (-10, 10)) -> np.ndarray:
        """
        随机旋转

        Args:
            image: 输入图像
            angle_range: 旋转角度范围

        Returns:
            旋转后的图像
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 应用旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        return rotated

    def random_brightness_contrast(self, image: np.ndarray,
                                  brightness_range: Tuple[float, float] = (-0.2, 0.2),
                                  contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        随机调整亮度和对比度

        Args:
            image: 输入图像
            brightness_range: 亮度调整范围
            contrast_range: 对比度调整范围

        Returns:
            调整后的图像
        """
        # 亮度调整
        brightness_delta = random.uniform(brightness_range[0], brightness_range[1])
        brightness_adjusted = np.clip(image.astype(np.float32) + brightness_delta * 255, 0, 255)

        # 对比度调整
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        contrast_adjusted = np.clip((brightness_adjusted - 128) * contrast_factor + 128, 0, 255)

        return contrast_adjusted.astype(np.uint8)

    def random_blur(self, image: np.ndarray, kernel_range: Tuple[int, int] = (1, 5)) -> np.ndarray:
        """
        随机模糊

        Args:
            image: 输入图像
            kernel_range: 模糊核大小范围

        Returns:
            模糊后的图像
        """
        kernel_size = random.choice([k for k in range(kernel_range[0], kernel_range[1] + 1) if k % 2 == 1])

        if kernel_size > 1:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return blurred

        return image

    def random_noise(self, image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        添加随机噪声

        Args:
            image: 输入图像
            noise_factor: 噪声强度

        Returns:
            添加噪声后的图像
        """
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)

        return noisy.astype(np.uint8)

    def mosaic_augmentation(self, images: List[np.ndarray], labels: List[List]) -> Tuple[np.ndarray, List]:
        """
        Mosaic数据增强

        Args:
            images: 4张输入图像列表
            labels: 对应的标签列表

        Returns:
            Mosaic增强后的图像和标签
        """
        if len(images) != 4:
            raise ValueError("Mosaic需要恰好4张图像")

        target_size = (self.img_size, self.img_size)
        mosaic_image = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)

        # 计算每个图像的位置
        center_x = random.randint(target_size[0] // 4, 3 * target_size[0] // 4)
        center_y = random.randint(target_size[1] // 4, 3 * target_size[1] // 4)

        # 四个图像的位置
        positions = [
            (0, 0, center_x, center_y),  # 左上
            (center_x, 0, target_size[0], center_y),  # 右上
            (0, center_y, center_x, target_size[1]),  # 左下
            (center_x, center_y, target_size[0], target_size[1])  # 右下
        ]

        combined_labels = []

        for i, (img, labels_i) in enumerate(zip(images, labels)):
            x1, y1, x2, y2 = positions[i]

            # 调整图像大小
            img_resized = cv2.resize(img, (x2 - x1, y2 - y1))

            # 放置到mosaic图像中
            mosaic_image[y1:y2, x1:x2] = img_resized

            # 调整标签坐标
            for label in labels_i:
                if len(label) >= 5:  # class_id, x_center, y_center, width, height
                    class_id = label[0]
                    x_center_norm = label[1]
                    y_center_norm = label[2]
                    width_norm = label[3]
                    height_norm = label[4]

                    # 转换为绝对坐标
                    img_h, img_w = img.shape[:2]
                    abs_x_center = x_center_norm * img_w
                    abs_y_center = y_center_norm * img_h
                    abs_width = width_norm * img_w
                    abs_height = height_norm * img_h

                    # 调整到mosaic坐标系
                    new_x_center = abs_x_center + x1
                    new_y_center = abs_y_center + y1

                    # 转换为归一化坐标
                    new_x_center_norm = new_x_center / target_size[0]
                    new_y_center_norm = new_y_center / target_size[1]
                    new_width_norm = abs_width / target_size[0]
                    new_height_norm = abs_height / target_size[1]

                    # 检查边界
                    if (0 < new_x_center_norm < 1 and 0 < new_y_center_norm < 1 and
                        new_x_center_norm - new_width_norm/2 > 0 and
                        new_x_center_norm + new_width_norm/2 < 1 and
                        new_y_center_norm - new_height_norm/2 > 0 and
                        new_y_center_norm + new_height_norm/2 < 1):

                        combined_labels.append([class_id, new_x_center_norm, new_y_center_norm,
                                              new_width_norm, new_height_norm])

        return mosaic_image, combined_labels

    def apply_transforms(self, image: np.ndarray, apply_augmentation: bool = True) -> np.ndarray:
        """
        应用数据增强

        Args:
            image: 输入图像
            apply_augmentation: 是否应用增强

        Returns:
            处理后的图像
        """
        # 调整大小
        processed = self.resize_and_pad(image)

        if apply_augmentation:
            # 随机应用增强
            if random.random() < 0.3:
                processed = self.random_rotation(processed)

            if random.random() < 0.5:
                processed = self.random_brightness_contrast(processed)

            if random.random() < 0.2:
                processed = self.random_blur(processed)

            if random.random() < 0.1:
                processed = self.random_noise(processed)

        return processed