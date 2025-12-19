import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pygame
import sys
import random

# =========================
# 可调参数（只改这里）
# =========================
TARGET_FPS = 30

COIN_COUNT = 2
OBSTACLE_COUNT = 2

# 速度（像素/秒）——你说偏快，我这里整体降了一档
SHIP_SPEED = 170          # 原来 220 -> 170（更稳）
COIN_SPEED = 190          # 原来 240 -> 190
OBSTACLE_SPEED_MIN = 170  # 原来 220 -> 170
OBSTACLE_SPEED_MAX = 280  # 原来 360 -> 280

INIT_LIVES = 5
GAME_SECONDS = 30

CONF_THRESHOLD = 0.5

# 防抖
ACTION_CONFIRM_FRAMES = 2
START_CONFIRM_FRAMES = 3
FIST_CONFIRM_FRAMES = 10   # 你已经改成 10

START_GESTURE = "ok"
QUIT_GESTURE = "fist"

# stop 不是暂停游戏，只是让飞机不动
GESTURE_TO_ACTION = {
    "like": -1,
    "dislike": 1,
    "stop": 0,
    "stop_inverted": 0,
    "four": 0
}

# 摄像头预览窗口
PREVIEW_W, PREVIEW_H = 200, 150
PREVIEW_MARGIN = 10

# =========================
# 模型
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
    out -= out[0]  # wrist
    scale = max(np.linalg.norm(out[9]), eps)
    return (out / scale).reshape(-1)

def standardize(feat, mean, scale, eps=1e-12):
    return (feat - mean) / (scale + eps)

def load_model(path):
    # bundle = torch.load(path, map_location="cpu")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(bundle["in_dim"], bundle["num_classes"], bundle["hidden"], bundle["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    scaler_mean = np.array(bundle["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(bundle["scaler_scale"], dtype=np.float32)
    labels = bundle["labels"]
    return model, scaler_mean, scaler_scale, labels

# =========================
# 生成对象
# =========================
def spawn_coin(W, H, offset_x=0):
    return {"x": float(W + offset_x), "y": float(random.randint(40, H - 40))}

def spawn_obstacle(W, H):
    r = random.randint(12, 22)
    return {
        "x": float(W + random.randint(0, 260)),
        "y": float(random.randint(40, H - 40)),
        "r": r,
        "v": float(random.randint(OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX)),
        "cooldown": 0
    }

def reset_round(W, H):
    st = {}
    st["ship_x"] = 80.0
    st["ship_y"] = float(H // 2)
    st["SHIP_HW"] = 30.0
    st["SHIP_HH"] = 18.0

    st["REWARD_R"] = 10.0
    st["coins"] = [spawn_coin(W, H, i * 220) for i in range(COIN_COUNT)]
    st["obstacles"] = [spawn_obstacle(W, H) for _ in range(OBSTACLE_COUNT)]

    st["score"] = 0
    st["lives"] = INIT_LIVES

    # 飞机动作（只影响飞机，不影响游戏）
    st["last_action"] = 0
    st["candidate_action"] = None
    st["candidate_count"] = 0

    return st

def action_name(a: int) -> str:
    return "UP" if a == -1 else ("DOWN" if a == 1 else "STOP")

# =========================
# 主程序
# =========================
def main():
    model, scaler_mean, scaler_scale, labels = load_model("saved_model/best_mlp_nosklearn.pt")

    # 标签一致性检查（避免“OK不生效”这种隐藏bug）
    if START_GESTURE not in labels:
        raise RuntimeError(f"模型 labels 中没有 '{START_GESTURE}'，请检查标签名。")
    if QUIT_GESTURE not in labels:
        raise RuntimeError(f"模型 labels 中没有 '{QUIT_GESTURE}'，请检查标签名。")

    pygame.init()
    WIDTH, HEIGHT = 640, 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Gesture Controlled Spaceship")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 28)
    big_font = pygame.font.SysFont(None, 48)

    ship_img = pygame.image.load("assets/ship.png").convert_alpha()
    coin_img = pygame.image.load("assets/coin.png").convert_alpha()
    ship_img = pygame.transform.scale(ship_img, (60, 36))
    coin_img = pygame.transform.scale(coin_img, (20, 20))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头：cv2.VideoCapture(0) 失败（可尝试改成 1）。")

    mp_hands = mp.solutions.hands

    # 状态：只有 WAIT_START / RUN / GAME_OVER（不再有“暂停整局”）
    st = reset_round(WIDTH, HEIGHT)
    state = "WAIT_START"
    start_ticks = None

    # 防抖计数
    ok_count = 0
    fist_count = 0

    # 显示用
    last_label = None
    last_conf = 0.0
    preview_surface = None

    with mp_hands.Hands(max_num_hands=1) as hands:
        running = True
        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0
            fps = clock.get_fps()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # ===== 摄像头 + 识别 =====
            label = None
            conf = 0.0
            ret, frame = cap.read()

            if ret:
                frame_flip = cv2.flip(frame, 1)

                # 右下角预览
                small = cv2.resize(frame_flip, (PREVIEW_W, PREVIEW_H))
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                preview_surface = pygame.surfarray.make_surface(np.transpose(rgb_small, (1, 0, 2)))

                # 手势识别
                rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lms = result.multi_hand_landmarks[0].landmark
                    lm_21x3 = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)

                    feat = preprocess_landmarks(lm_21x3)
                    feat = standardize(feat, scaler_mean, scaler_scale).astype(np.float32)
                    feat = torch.from_numpy(feat).unsqueeze(0)

                    with torch.no_grad():
                        prob = torch.softmax(model(feat), dim=1).cpu().numpy()[0]

                    idx = int(prob.argmax())
                    conf = float(prob[idx])
                    if conf >= CONF_THRESHOLD:
                        label = labels[idx]

            last_label, last_conf = label, conf

            # ===== OK 开始 / 重开（防抖）=====
            def ok_confirmed():
                nonlocal ok_count
                if label == START_GESTURE:
                    ok_count += 1
                else:
                    ok_count = 0
                return ok_count >= START_CONFIRM_FRAMES

            # ===== FIST 退出（只允许在：WAIT_START / GAME_OVER / 飞机STOP时）=====
            # 你要求“退出只在飞机停止时可用”，这里严格按 last_action==0 控制
            allow_quit = (state in ("WAIT_START", "GAME_OVER")) or (st["last_action"] == 0)

            if allow_quit and label == QUIT_GESTURE:
                fist_count += 1
                if fist_count >= FIST_CONFIRM_FRAMES:
                    break
            else:
                fist_count = 0

            # ===== 状态机 =====
            if state == "WAIT_START":
                # 开始前整局不动（但画面在）
                if ok_confirmed():
                    st = reset_round(WIDTH, HEIGHT)
                    start_ticks = pygame.time.get_ticks()
                    state = "RUN"
                    ok_count = 0

            elif state == "RUN":
                elapsed = (pygame.time.get_ticks() - start_ticks) / 1000.0
                remaining = max(0, int(GAME_SECONDS - elapsed))

                if remaining <= 0 or st["lives"] <= 0:
                    state = "GAME_OVER"
                    ok_count = 0
                else:
                    # 1) 更新飞机动作（只影响飞机）
                    if label in GESTURE_TO_ACTION:
                        new_action = GESTURE_TO_ACTION[label]
                        if new_action == st["last_action"]:
                            st["candidate_action"] = None
                            st["candidate_count"] = 0
                        else:
                            if st["candidate_action"] == new_action:
                                st["candidate_count"] += 1
                            else:
                                st["candidate_action"] = new_action
                                st["candidate_count"] = 1

                            if st["candidate_count"] >= ACTION_CONFIRM_FRAMES:
                                st["last_action"] = new_action
                                st["candidate_action"] = None
                                st["candidate_count"] = 0

                    # 2) 飞机位置 dt 更新
                    st["ship_y"] += st["last_action"] * SHIP_SPEED * dt
                    st["ship_y"] = np.clip(st["ship_y"], st["SHIP_HH"], HEIGHT - st["SHIP_HH"])

                    # 3) 金币 dt 更新（游戏一直跑）
                    for coin in st["coins"]:
                        coin["x"] -= COIN_SPEED * dt
                        if coin["x"] < 0:
                            coin["x"] = float(WIDTH)
                            coin["y"] = float(random.randint(40, HEIGHT - 40))

                        if (abs(st["ship_x"] - coin["x"]) < st["SHIP_HW"] + st["REWARD_R"] and
                            abs(st["ship_y"] - coin["y"]) < st["SHIP_HH"] + st["REWARD_R"]):
                            st["score"] += 1
                            coin["x"] = float(WIDTH)
                            coin["y"] = float(random.randint(40, HEIGHT - 40))

                    # 4) 障碍 dt 更新（游戏一直跑）
                    for ob in st["obstacles"]:
                        ob["x"] -= ob["v"] * dt
                        if ob["x"] < -60:
                            ob.update(spawn_obstacle(WIDTH, HEIGHT))

                        if ob["cooldown"] > 0:
                            ob["cooldown"] -= 1

                        if ob["cooldown"] == 0:
                            if (abs(st["ship_x"] - ob["x"]) < st["SHIP_HW"] + ob["r"] and
                                abs(st["ship_y"] - ob["y"]) < st["SHIP_HH"] + ob["r"]):
                                st["lives"] -= 1
                                ob["cooldown"] = 15
                                ob.update(spawn_obstacle(WIDTH, HEIGHT))

            elif state == "GAME_OVER":
                # 结束后 OK 重开
                if ok_confirmed():
                    st = reset_round(WIDTH, HEIGHT)
                    start_ticks = pygame.time.get_ticks()
                    state = "RUN"
                    ok_count = 0

            # ===== 绘制 =====
            screen.fill((10, 10, 30))

            # 飞船
            screen.blit(
                ship_img,
                (int(st["ship_x"] - ship_img.get_width() // 2),
                 int(st["ship_y"] - ship_img.get_height() // 2))
            )

            # 金币 / 障碍
            for coin in st["coins"]:
                screen.blit(
                    coin_img,
                    (int(coin["x"] - coin_img.get_width() // 2),
                     int(coin["y"] - coin_img.get_height() // 2))
                )
            for ob in st["obstacles"]:
                pygame.draw.circle(screen, (200, 80, 80), (int(ob["x"]), int(ob["y"])), int(ob["r"]))

            # HUD
            screen.blit(font.render(f"Lives: {st['lives']}", True, (255, 255, 255)), (10, 10))
            screen.blit(font.render(f"Score: {st['score']}", True, (255, 255, 255)), (10, 38))

            if state == "RUN" and start_ticks is not None:
                elapsed = (pygame.time.get_ticks() - start_ticks) / 1000.0
                remaining = max(0, int(GAME_SECONDS - elapsed))
            else:
                remaining = GAME_SECONDS
            screen.blit(font.render(f"Time: {remaining}s", True, (255, 255, 255)), (10, 66))
            screen.blit(font.render(f"FPS: {fps:.1f}", True, (255, 255, 0)), (10, 94))

            # 右下角预览 + label/action
            if preview_surface is not None:
                px = WIDTH - PREVIEW_W - PREVIEW_MARGIN
                py = HEIGHT - PREVIEW_H - PREVIEW_MARGIN
                screen.blit(preview_surface, (px, py))
                pygame.draw.rect(screen, (255, 255, 255), (px, py, PREVIEW_W, PREVIEW_H), 2)

                info1 = font.render(f"label: {str(last_label)} ({last_conf:.2f})", True, (255, 255, 255))
                info2 = font.render(f"action: {action_name(st['last_action'])}", True, (255, 255, 255))
                info3 = font.render(f"quit_ok: {allow_quit}", True, (180, 180, 180))
                screen.blit(info1, (px, py - 60))
                screen.blit(info2, (px, py - 40))
                screen.blit(info3, (px, py - 20))

            # 文案提示
            if state == "WAIT_START":
                msg = big_font.render("Show 'OK' to start", True, (255, 255, 0))
                screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))
                tip = font.render("STOP = ship stop (game continues)", True, (255, 255, 255))
                screen.blit(tip, (WIDTH // 2 - tip.get_width() // 2, HEIGHT // 2 + 18))

            elif state == "GAME_OVER":
                msg = big_font.render("GAME OVER", True, (255, 255, 0))
                screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 50))
                msg2 = font.render(f"Final Score: {st['score']}", True, (255, 255, 255))
                screen.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, HEIGHT // 2 - 10))
                msg3 = font.render("Show 'OK' to play again", True, (255, 255, 255))
                screen.blit(msg3, (WIDTH // 2 - msg3.get_width() // 2, HEIGHT // 2 + 20))
                msg4 = font.render("FIST quits only when ship STOP", True, (200, 200, 200))
                screen.blit(msg4, (WIDTH // 2 - msg4.get_width() // 2, HEIGHT // 2 + 45))

            pygame.display.flip()

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
