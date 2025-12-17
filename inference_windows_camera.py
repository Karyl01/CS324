import subprocess
import numpy as np
import cv2
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
    out = out - out[0].copy()              # wrist(0) 为原点
    scale = max(np.linalg.norm(out[9]), eps)  # wrist->middle_mcp(9)
    out = out / scale
    return out.reshape(-1)

def standardize(feat_1d: np.ndarray, mean: np.ndarray, scale: np.ndarray, eps=1e-12) -> np.ndarray:
    return (feat_1d - mean) / (scale + eps)

def load_model(bundle_path: str):
    bundle = torch.load(bundle_path, map_location="cpu")

    model = MLP(
        in_dim=bundle["in_dim"],
        num_classes=bundle["num_classes"],
        hidden=bundle["hidden"],
        dropout=bundle["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    scaler_mean = np.array(bundle["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(bundle["scaler_scale"], dtype=np.float32)
    labels = bundle["labels"]
    return model, scaler_mean, scaler_scale, labels

# =========================
# 2. 用 ffmpeg 从 RTSP 拉流
# =========================
def open_rtsp_ffmpeg(rtsp_url: str, W: int, H: int):
    cmd = [
        "ffmpeg",
        "-loglevel", "error",          # 少点输出，必要时改成 info/debug
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-an",
        "-vf", f"scale={W}:{H}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    return p

def read_frame_ffmpeg(p, W: int, H: int):
    need = W * H * 3
    raw = p.stdout.read(need)
    if len(raw) != need:
        return None
    frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
    return frame

# =========================
# 3. 主程序
# =========================
def main():
    MODEL_PATH = "saved_model/best_mlp_nosklearn.pt"
    CONF_THRESHOLD = 0.7

    # RTSP
    rtsp_url = "rtsp://10.28.108.164:8554/live"
    W, H = 640, 480

    model, scaler_mean, scaler_scale, labels = load_model(MODEL_PATH)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    p = open_rtsp_ffmpeg(rtsp_url, W, H)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        pred_text = "No hand"
        while True:
            frame = read_frame_ffmpeg(p, W, H)
            if frame is None:
                print("RTSP 断流/读不到帧，退出。")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 把上一次的预测结果也叠在画面上
            cv2.putText(frame, pred_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Press 'p' to predict, 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Gesture Recognition (RTSP)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0].landmark
                lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                feat = preprocess_landmarks(lm_21x3)
                feat = standardize(feat, scaler_mean, scaler_scale).astype(np.float32).reshape(1, -1)

                with torch.no_grad():
                    logits = model(torch.from_numpy(feat))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

                idx = int(prob.argmax())
                conf = float(prob[idx])

                pred_text = f"{labels[idx]} ({conf:.2f})" if conf >= CONF_THRESHOLD else f"none ({conf:.2f})"
                print("Prediction:", pred_text)

            if key == ord("q"):
                break

    # 清理
    try:
        p.terminate()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
