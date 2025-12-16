import os
import glob
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp


@dataclass
class PreprocessConfig:
    choose_largest_hand: bool = True
    center_on_wrist: bool = True
    scale_by_middle_mcp: bool = True
    eps: float = 1e-6
    flatten: bool = True
    skip_no_hand: bool = True


def list_images_by_label(dataset_root: str) -> List[Tuple[str, str]]:
    pairs = []
    for label in sorted(os.listdir(dataset_root)):
        label_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(label_dir):
            continue

        exts = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(label_dir, f"*.{ext}")))

        # ✅ 保证“从头开始”稳定：排序
        img_paths = sorted(img_paths)

        for p in img_paths:
            pairs.append((p, label))
    return pairs


def print_raw_dataset_stats(pairs: List[Tuple[str, str]]) -> None:
    labels = [lb for _, lb in pairs]
    counter = Counter(labels)

    classes = sorted(counter.keys())
    total_images = len(pairs)
    counts = [counter[c] for c in classes] if classes else []

    print("\n========== Raw Dataset Summary ==========")
    print(f"Total classes: {len(classes)}")
    print(f"Total images: {total_images}\n")

    print("Per-class image counts:")
    print("-" * 55)
    print(f"{'Class':<25}{'Images':>10}")
    print("-" * 55)
    for cls in classes:
        print(f"{cls:<25}{counter[cls]:>10}")
    print("-" * 55)

    if counts:
        min_c, max_c = min(counts), max(counts)
        ratio = max_c / max(min_c, 1)
        print(f"Min images in class: {min_c}")
        print(f"Max images in class: {max_c}")
        print(f"Imbalance ratio (max/min): {ratio:.2f}")

    print("========================================\n")


def limit_pairs_per_class(pairs: List[Tuple[str, str]], per_class_limit: int) -> List[Tuple[str, str]]:
    """
    ✅ 每个类别只保留前 per_class_limit 个样本（从头开始）。
    per_class_limit = 0 表示不限制（全量）。
    """
    if per_class_limit <= 0:
        return pairs

    kept = []
    counts = defaultdict(int)
    for img_path, label in pairs:
        if counts[label] < per_class_limit:
            kept.append((img_path, label))
            counts[label] += 1
    return kept


def hand_bbox_area(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def preprocess_landmarks(lm_21x3: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    out = lm_21x3.astype(np.float32)

    if cfg.center_on_wrist:
        wrist = out[0].copy()
        out = out - wrist

    if cfg.scale_by_middle_mcp:
        scale = np.linalg.norm(out[9] - out[0]) if not cfg.center_on_wrist else np.linalg.norm(out[9])
        scale = max(scale, cfg.eps)
        out = out / scale

    if cfg.flatten:
        out = out.reshape(-1)

    return out


def extract_one_image_features(
    img_path: str,
    hands: mp.solutions.hands.Hands,
    cfg: PreprocessConfig,
) -> Tuple[Optional[np.ndarray], Dict]:
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None, {"error": "imread_failed"}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, {"error": "no_hand"}

    idx = 0
    if cfg.choose_largest_hand and len(result.multi_hand_landmarks) > 1:
        areas = [hand_bbox_area(h) for h in result.multi_hand_landmarks]
        idx = int(np.argmax(areas))

    hand_lms = result.multi_hand_landmarks[idx].landmark
    lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

    feat = preprocess_landmarks(lm_21x3, cfg)

    meta = {"num_hands": len(result.multi_hand_landmarks), "chosen_hand_index": idx}

    if result.multi_handedness and len(result.multi_handedness) > idx:
        meta["handedness"] = result.multi_handedness[idx].classification[0].label
        meta["handedness_score"] = float(result.multi_handedness[idx].classification[0].score)

    return feat, meta


def print_extraction_stats(stats: Dict) -> None:
    print("\n========== Extraction Summary (After MediaPipe) ==========")
    print(f"Total images scanned: {stats['total_images']}")
    print(f"Total valid samples:  {stats['total_valid']}")
    print(f"Total skipped:        {stats['total_skipped']}\n")

    print("Per-class statistics:")
    print("-" * 60)
    print(f"{'Class':<20}{'Total':>10}{'Valid':>10}{'Skipped':>12}")
    print("-" * 60)

    valid_counts = []
    for cls in sorted(stats["per_class"].keys()):
        s = stats["per_class"][cls]
        print(f"{cls:<20}{s['total']:>10}{s['valid']:>10}{s['skipped']:>12}")
        valid_counts.append(s["valid"])

    print("-" * 60)
    if valid_counts:
        min_v, max_v = min(valid_counts), max(valid_counts)
        ratio = max_v / max(min_v, 1)
        print(f"Min valid samples in class: {min_v}")
        print(f"Max valid samples in class: {max_v}")
        print(f"Imbalance ratio (max/min):  {ratio:.2f}")

    print("=========================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/hagrid classification 512p/hagrid-classification-512p",
        help="HaGRID 数据集根目录（里面是各类别文件夹）",
    )
    parser.add_argument("--out_npz", type=str, default="data/output_features/features.npz", help="输出 npz 文件路径")
    parser.add_argument("--out_csv", type=str, default="data/output_features/features.csv", help="输出 csv 文件路径（留空则不输出）")

    # ✅ 新增：每类最多处理多少张（0=全量）
    parser.add_argument("--per_class_limit", type=int, default=1000, help="每个类别最多提取多少张图片（0=全量，从头开始）")
    # 原来的全局调试截断（可保留）
    parser.add_argument("--max_images", type=int, default=0, help="调试用：最多处理多少张（0=全量）")

    # MediaPipe Hands 参数（静态手势建议）
    parser.add_argument("--static_image_mode", action="store_true", help="静态图模式（更适合图片）")
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    # 预处理开关
    parser.add_argument("--no_center", action="store_true")
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--no_flatten", action="store_true")
    parser.add_argument("--keep_no_hand", action="store_true", help="遇到无手图片不跳过，而是写 NaN")

    args = parser.parse_args()

    cfg = PreprocessConfig(
        center_on_wrist=not args.no_center,
        scale_by_middle_mcp=not args.no_scale,
        flatten=not args.no_flatten,
        skip_no_hand=not args.keep_no_hand,
    )

    # 1) 扫描原始数据集并统计（全量）
    pairs_all = list_images_by_label(args.dataset_root)
    if not pairs_all:
        raise RuntimeError(f"在 {args.dataset_root} 下没有找到任何图片。请确认目录结构是 dataset_root/label/*.jpg")

    print_raw_dataset_stats(pairs_all)

    # 2) ✅ 每类取前 N 张（如果 per_class_limit=0 则不限制）
    pairs = limit_pairs_per_class(pairs_all, args.per_class_limit)
    if args.per_class_limit > 0:
        print(f"[Info] per_class_limit={args.per_class_limit} enabled -> will process up to {args.per_class_limit} images per class.")
        print(f"[Info] Total images after per-class limit: {len(pairs)}\n")

    # 3) 全局截断（调试用）
    if args.max_images and args.max_images > 0:
        pairs = pairs[: args.max_images]
        print(f"[Info] max_images enabled -> only process first {len(pairs)} images after filtering.\n")

    # label 编码（基于当前要处理的 pairs）
    labels = sorted(list({lb for _, lb in pairs}))
    label2id = {lb: i for i, lb in enumerate(labels)}

    # 4) 初始化提取统计
    stats = {
        "total_images": 0,
        "total_valid": 0,
        "total_skipped": 0,
        "per_class": defaultdict(lambda: {"total": 0, "valid": 0, "skipped": 0}),
    }

    X_list, y_list, path_list, meta_list = [], [], [], []

    # 5) MediaPipe 提取 landmarks
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=args.static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=1,
    ) as hands:

        for img_path, label in tqdm(pairs, desc="Extract landmarks"):
            stats["total_images"] += 1
            stats["per_class"][label]["total"] += 1

            feat, meta = extract_one_image_features(img_path, hands, cfg)

            if feat is None:
                stats["total_skipped"] += 1
                stats["per_class"][label]["skipped"] += 1

                if cfg.skip_no_hand:
                    continue

                feat = np.full((63,), np.nan, dtype=np.float32) if cfg.flatten else np.full((21, 3), np.nan, dtype=np.float32)
            else:
                stats["total_valid"] += 1
                stats["per_class"][label]["valid"] += 1

            X_list.append(feat)
            y_list.append(label2id[label])
            path_list.append(img_path)
            meta_list.append(meta)

    if not X_list:
        raise RuntimeError("没有得到任何有效样本（全部 no_hand 或读图失败）。请调低检测阈值/检查数据。")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    paths = np.array(path_list, dtype=object)

    # 6) 保存 NPZ
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X=X,
        y=y,
        paths=paths,
        labels=np.array(labels, dtype=object),
        label2id=np.array(json.dumps(label2id), dtype=object),
    )
    print(f"[OK] Saved NPZ: {args.out_npz}")
    print(f"     X shape={X.shape}, y shape={y.shape}, num_labels={len(labels)}")

    # 7) 保存 CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])] if cfg.flatten else None)
        df["y"] = y
        df["label"] = [labels[i] for i in y]
        df["path"] = paths
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] Saved CSV: {args.out_csv}")

    # 8) 输出 MediaPipe 提取后的统计
    print_extraction_stats(stats)


if __name__ == "__main__":
    main()
