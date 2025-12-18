import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pygame
import sys

# =========================
# 手势 → 动作（上下）
# =========================
GESTURE_TO_ACTION = {
    "like": -1,  # 向上
    "dislike": 1,            # 向下
    "stop": 0,              # 停止
    "stop_inverted": 0,
    "four": 0
}

# =========================
# 模型定义（不变）
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

def preprocess_landmarks(lm_21x3, eps=1e-6):
    out = lm_21x3.astype(np.float32)
    out -= out[0]
    scale = max(np.linalg.norm(out[9]), eps)
    return (out / scale).reshape(-1)

def standardize(feat, mean, scale, eps=1e-12):
    return (feat - mean) / (scale + eps)

def load_model(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(bundle["in_dim"], bundle["num_classes"],
                bundle["hidden"], bundle["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, np.array(bundle["scaler_mean"]), np.array(bundle["scaler_scale"]), bundle["labels"]

# =========================
# 主程序
# =========================
def main():
    # -------- 模型 --------
    model, scaler_mean, scaler_scale, labels = load_model(
        "saved_model/best_mlp_nosklearn.pt"
    )
    CONF_THRESHOLD = 0.5

    # -------- pygame 初始化 --------
    pygame.init()

    WIDTH, HEIGHT = 640, 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    ship_img = pygame.image.load("assets/ship.png").convert_alpha()
    reward_img = pygame.image.load("assets/coin.png").convert_alpha()
    # bg_img = pygame.image.load("assets/background.png").convert()

    ship_img = pygame.transform.scale(ship_img, (60, 36))
    reward_img = pygame.transform.scale(reward_img, (20, 20))

    pygame.display.set_caption("Gesture Controlled Spaceship")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    # -------- 游戏状态 --------
    ship_x = 80
    ship_y = HEIGHT // 2
    ship_speed = 4

    SHIP_HW, SHIP_HH = 30, 18
    REWARD_R = 10

    reward_x = WIDTH
    reward_y = np.random.randint(40, HEIGHT - 40)
    reward_speed = 4

    score = 0

    last_action = 0
    candidate_action = None
    candidate_count = 0
    DEBOUNCE_FRAMES = 2

    # -------- 摄像头 & mediapipe（后台）--------
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(max_num_hands=1) as hands:
        running = True
        while running:
            # ===== pygame 事件 =====
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # ===== 摄像头读取（不显示）=====
            ret, frame = cap.read()
            label = None

            if ret:
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lms = result.multi_hand_landmarks[0].landmark
                    lm_21x3 = np.array([[lm.x, lm.y, lm.z] for lm in lms])
                    feat = preprocess_landmarks(lm_21x3)
                    feat = standardize(feat, scaler_mean, scaler_scale)
                    feat = torch.from_numpy(feat).unsqueeze(0)

                    with torch.no_grad():
                        prob = torch.softmax(model(feat), 1)[0]
                    idx = int(prob.argmax())

                    print(labels[idx], prob[idx])

                    if prob[idx] > CONF_THRESHOLD:
                        label = labels[idx]

            # ===== 防抖动作切换 =====
            if label in GESTURE_TO_ACTION:
                new_action = GESTURE_TO_ACTION[label]
                if new_action == last_action:
                    candidate_action = None
                    candidate_count = 0
                else:
                    if candidate_action == new_action:
                        candidate_count += 1
                    else:
                        candidate_action = new_action
                        candidate_count = 1
                    if candidate_count >= DEBOUNCE_FRAMES:
                        last_action = new_action
                        candidate_action = None
                        candidate_count = 0

            # ===== 更新飞船 =====
            ship_y += last_action * ship_speed
            ship_y = np.clip(ship_y, SHIP_HH, HEIGHT - SHIP_HH)

            # ===== 更新奖励 =====
            reward_x -= reward_speed
            if reward_x < 0:
                reward_x = WIDTH
                reward_y = np.random.randint(40, HEIGHT - 40)

            if (abs(ship_x - reward_x) < SHIP_HW + REWARD_R and
                abs(ship_y - reward_y) < SHIP_HH + REWARD_R):
                score += 1
                reward_x = WIDTH
                reward_y = np.random.randint(40, HEIGHT - 40)

            # ===== pygame 绘制 =====
            screen.fill((10, 10, 30))

            screen.blit(
                ship_img,
                (ship_x - ship_img.get_width() // 2,
                 ship_y - ship_img.get_height() // 2)
            )

            screen.blit(
                reward_img,
                (reward_x - reward_img.get_width() // 2,
                 reward_y - reward_img.get_height() // 2)
            )

            score_text = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))

            pygame.display.flip()
            clock.tick(30)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
