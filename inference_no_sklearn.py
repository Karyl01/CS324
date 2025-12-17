import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# =========================
# 1. MLP 模型（必须与训练一致）
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.2):
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

def preprocess_landmarks(lm_21x3: np.ndarray, eps=1e-6) -> np.ndarray:
    out = lm_21x3.astype(np.float32)
    wrist = out[0].copy()
    out = out - wrist
    scale = np.linalg.norm(out[9])
    scale = max(scale, eps)
    out = out / scale
    return out.reshape(-1)

def standardize(feat_1d: np.ndarray, mean: np.ndarray, scale: np.ndarray, eps=1e-12) -> np.ndarray:
    # feat_1d: (63,)
    return (feat_1d - mean) / (scale + eps)

def load_model(bundle_path: str):
    # bundle = torch.load(bundle_path, map_location="cpu", weights_only=False) # 这一行在windows上运行
    bundle = torch.load(bundle_path, map_location="cpu") # 这一行在开发板运行

    model = MLP(
        in_dim=bundle["in_dim"],
        num_classes=bundle["num_classes"],
        hidden=bundle["hidden"],
        dropout=bundle["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    # 从 bundle 里取纯 numpy 参数
    scaler_mean = np.array(bundle["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(bundle["scaler_scale"], dtype=np.float32)
    labels = bundle["labels"]
    return model, scaler_mean, scaler_scale, labels

def main():
    MODEL_PATH = "saved_model/best_mlp_nosklearn.pt"
    CONF_THRESHOLD = 0.7

    model, scaler_mean, scaler_scale, labels = load_model(MODEL_PATH)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Press 'p' to predict, 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0].landmark
                lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                feat = preprocess_landmarks(lm_21x3)  # (63,)
                feat = standardize(feat, scaler_mean, scaler_scale).astype(np.float32)
                feat = feat.reshape(1, -1)

                with torch.no_grad():
                    logits = model(torch.from_numpy(feat))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

                idx = int(prob.argmax())
                conf = float(prob[idx])
                pred_text = f"{labels[idx]} ({conf:.2f})" if conf >= CONF_THRESHOLD else f"none ({conf:.2f})"
                print("Prediction:", pred_text)

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
