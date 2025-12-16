import argparse
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_labels_mapping(df: pd.DataFrame):
    if "label" in df.columns:
        y_to_label = (
            df[["y", "label"]]
            .drop_duplicates()
            .sort_values("y")
            .set_index("y")["label"]
            .to_dict()
        )
        labels = [y_to_label[i] for i in sorted(y_to_label.keys())]
    else:
        labels = [str(i) for i in sorted(df["y"].unique())]
    return labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_y = []
    all_prob = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)
        all_prob.append(prob)
        all_pred.append(pred)
        all_y.append(yb.numpy())
    all_pred = np.concatenate(all_pred)
    all_y = np.concatenate(all_y)
    all_prob = np.concatenate(all_prob)
    return all_y, all_pred, all_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="features.csv 路径", default='data/output_features/features.csv')
    ap.add_argument("--out", default="saved_model/best_mlp.pt", help="保存 best 模型（包含 scaler/labels）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)

    ap.add_argument("--patience", type=int, default=10, help="验证集 early stopping patience")
    ap.add_argument("--max_rows", type=int, default=0, help="调试用：只读前 N 行，0=全量")
    args = ap.parse_args()

    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1) 读 CSV
    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()

    feature_cols = [c for c in df.columns if c.startswith("f")]
    if len(feature_cols) != 63:
        raise ValueError(f"期望 63 个特征列 f0..f62，但检测到 {len(feature_cols)} 个。")

    X = df[feature_cols].to_numpy(np.float32)
    y = df["y"].to_numpy(np.int64)

    # label 映射（用来输出报告）
    labels = get_labels_mapping(df)

    # 2) 清洗 NaN
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # 3) 三划分 train/val/test（分层）
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(args.val_size + args.test_size),
        random_state=args.seed,
        stratify=y
    )
    val_ratio_in_tmp = args.val_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1 - val_ratio_in_tmp),
        random_state=args.seed,
        stratify=y_tmp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    num_classes = int(np.max(y) + 1)
    print(f"Num classes: {num_classes}")

    # 4) 标准化（只用 train 拟合）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 5) DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch, shuffle=False
    )

    # 6) 模型/优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=X_train.shape[1], num_classes=num_classes, hidden=args.hidden, dropout=args.dropout).to(device)

    # 类别不均衡时更稳：加权 CrossEntropy（可选）
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = (class_counts.sum() / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度（可选但很有用）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 7) 训练 + early stopping（按 val accuracy 保存 best）
    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

        # 验证
        y_true, y_pred, _ = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)

        scheduler.step(val_acc)

        print(f"Epoch {epoch:03d} | train_loss={total_loss/total:.4f} | val_acc={val_acc:.4f}")

        # 保存 best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"  -> best updated (val_acc={best_val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    # 8) 载入 best，测试集评估
    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    y_true, y_pred, _ = evaluate(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\n[TEST] acc={test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # 9) 保存 best checkpoint（包含 scaler + labels）
    bundle = {
        "state_dict": model.state_dict(),
        "in_dim": int(X_train.shape[1]),
        "num_classes": int(num_classes),
        "hidden": int(args.hidden),
        "dropout": float(args.dropout),
        "labels": labels,
        "scaler": scaler,  # sklearn 对象，推理时必须用同一个
        "feature_cols": feature_cols,
        "best_val_acc": float(best_val_acc),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(bundle, args.out)
    print(f"\n[OK] Saved best model to: {args.out} (best_val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
