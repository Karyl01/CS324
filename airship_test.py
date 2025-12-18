#测试版本，可以查看手势

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

GESTURE_TO_ACTION = {
    "like": -1,  # 向上
    "dislike": 1,            # 向下
    "stop": 0,              # 停止
    "stop_inverted": 0
}

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
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False) # 这一行在windows上运行
    # bundle = torch.load(bundle_path, map_location="cpu") # 这一行在开发板运行

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

    ship_x = 80  # 靠左，但留一点边距
    ship_y = 240  # 屏幕垂直中间（480 / 2）

    ship_speed = 8

    reward_x = 640  # 从屏幕右侧出现
    reward_y = np.random.randint(40, 460)
    reward_speed = 5

    score = 0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    last_action = 0  # 当前生效的动作
    candidate_action = None  # 正在尝试切换的动作
    candidate_count = 0  # 连续出现次数

    DEBOUNCE_FRAMES = 3  # 连续 3 帧确认

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

            cv2.putText(frame, "Gesture Control Spaceship",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF

            label = None

            if result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0].landmark
                lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                feat = preprocess_landmarks(lm_21x3)
                feat = standardize(feat, scaler_mean, scaler_scale).astype(np.float32)
                feat = feat.reshape(1, -1)

                with torch.no_grad():
                    logits = model(torch.from_numpy(feat))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

                idx = int(prob.argmax())
                conf = float(prob[idx])

                print(labels[idx], conf)

                if conf >= CONF_THRESHOLD:
                    label = labels[idx]

            # =========================
            # 飞船控制（带记忆）
            # =========================
            if label in GESTURE_TO_ACTION:
                new_action = GESTURE_TO_ACTION[label]

                if new_action == last_action:
                    # 和当前一致，直接重置候选
                    candidate_action = None
                    candidate_count = 0

                else:
                    # 尝试切换动作
                    if candidate_action == new_action:
                        candidate_count += 1
                    else:
                        candidate_action = new_action
                        candidate_count = 1

                    # 连续多帧确认，正式切换
                    if candidate_count >= DEBOUNCE_FRAMES:
                        last_action = new_action
                        candidate_action = None
                        candidate_count = 0

            ship_y += last_action * ship_speed
            ship_y = np.clip(ship_y, 40, 460)

            # =========================
            # 奖励逻辑
            # =========================
            reward_x -= reward_speed

            if reward_x < 0:
                reward_x = 640
                reward_y = np.random.randint(40, 460)

            # 碰撞检测
            if abs(ship_x - reward_x) < 40 and abs(ship_y - reward_y) < 28:
                score += 1
                reward_x = 640  # 从右侧重生
                reward_y = np.random.randint(40, 460)

            # =========================
            # 画游戏元素
            # =========================
            # 飞船
            cv2.rectangle(frame,
                          (ship_x - 30, ship_y - 18),
                          (ship_x + 30, ship_y + 18),
                          (255, 255, 0), -1)

            # 奖励
            cv2.circle(frame,
                       (reward_x, reward_y),
                       10, (0, 255, 0), -1)

            # 分数
            cv2.putText(frame, f"Score: {score}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

            cv2.imshow("Gesture Recognition", frame)

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()