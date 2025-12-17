import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

import random

KEY_TO_GESTURE = {
    ord('1'): 'fist',    # 石头
    ord('2'): 'peace',   # 剪刀
    ord('3'): 'palm',    # 布
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


# =========================
# 2. 与训练时完全一致的预处理
# =========================
def preprocess_landmarks(lm_21x3: np.ndarray, eps=1e-6) -> np.ndarray:
    """
    lm_21x3: (21,3) MediaPipe normalized landmarks
    返回: (63,) 预处理后的特征
    """
    out = lm_21x3.astype(np.float32)

    # (1) 以 wrist(0) 为原点
    wrist = out[0].copy()
    out = out - wrist

    # (2) 用 wrist -> middle_mcp(9) 归一化
    scale = np.linalg.norm(out[9])
    scale = max(scale, eps)
    out = out / scale

    return out.reshape(-1)  # (63,)


# =========================
# 3. 加载模型
# =========================
def load_model(bundle_path: str):
    torch.serialization.add_safe_globals([StandardScaler])

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    model = MLP(
        in_dim=bundle["in_dim"],
        num_classes=bundle["num_classes"],
        hidden=bundle["hidden"],
        dropout=bundle["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    scaler = bundle["scaler"]
    labels = bundle["labels"]
    return model, scaler, labels


# =========================
# 4. 主程序：摄像头实时推理
# =========================
def main():
    MODEL_PATH = 'saved_model/best_mlp.pt'   # ← 改成你的模型路径
    CONF_THRESHOLD = 0.7         # 置信度阈值

    model, scaler, labels = load_model(MODEL_PATH)

    player_gesture = None
    computer_gesture = None
    game_result = None

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print("按键说明：")
    print("  q  : 退出")
    print("  p  : 拍照并进行一次手势预测")

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

            pred_text = "No hand"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # 画手部骨架
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # 显示每个点的坐标（x,y,z）
                h, w, _ = frame.shape
                for i, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    text = f"{i}"
                    cv2.putText(frame, text, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 显示提示
            cv2.putText(
                frame,
                "Press 'p' to predict, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            y0 = 70
            cv2.putText(frame, "Rock Paper Scissors", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if player_gesture:
                cv2.putText(frame, f"Player: {player_gesture}", (10, y0 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if computer_gesture:
                cv2.putText(frame, f"Computer: {computer_gesture}", (10, y0 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if game_result:
                cv2.putText(frame, f"Result: {game_result}", (10, y0 + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            # ====== 键盘模拟手势（无摄像头调试用）======
            if key in KEY_TO_GESTURE:
                player_gesture = KEY_TO_GESTURE[key]
                computer_gesture = random.choice(["fist", "peace", "palm"])
                game_result = judge_rps(player_gesture, computer_gesture)

            # ====== 拍照并预测 ======
            if key == ord("p") and result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0].landmark
                lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                feat = preprocess_landmarks(lm_21x3)
                feat = scaler.transform(feat.reshape(1, -1)).astype(np.float32)

                with torch.no_grad():
                    logits = model(torch.from_numpy(feat))
                    prob = torch.softmax(logits, dim=1).numpy()[0]

                idx = int(prob.argmax())
                conf = float(prob[idx])

                label = labels[idx]

                if conf >= CONF_THRESHOLD and label in ["fist", "peace", "palm"]:
                    player_gesture = label
                    computer_gesture = random.choice(["fist", "peace", "palm"])
                    game_result = judge_rps(player_gesture, computer_gesture)
                else:
                    player_gesture = None
                    game_result = "Unrecognized"

                print("player_gesture = ", player_gesture)
                print("computer_gesture = ", computer_gesture)
                print("game_result = ", game_result)

            # ====== 退出 ======
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# 判断逻辑
def judge_rps(player, computer):
    if player == computer:
        return "Draw"
    if (player == "fist" and computer == "peace") or \
       (player == "peace" and computer == "palm") or \
       (player == "palm" and computer == "fist"):
        return "You Win"
    return "Computer Wins"



if __name__ == "__main__":
    main()
