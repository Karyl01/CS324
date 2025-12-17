import subprocess
import numpy as np
import cv2

W, H = 640, 480
rtsp_url = "rtsp://10.28.108.164:8554/live"

cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", rtsp_url,
    "-an",
    "-vf", f"scale={W}:{H}",
    "-pix_fmt", "bgr24",
    "-f", "rawvideo",
    "pipe:1",
]

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W*H*3*2)

while True:
    raw = p.stdout.read(W*H*3)
    if len(raw) != W*H*3:
        break
    frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
