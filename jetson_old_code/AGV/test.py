# ==============================================================
# client_opencv.py
# Bám làn bằng OpenCV (double-yellow, fallback vạch trắng) + Stop/Go (YOLO)
# - KHÔNG dùng DeepLab
# - Fix JSON "Extra data" bằng buffer + raw_decode
# - Ép int khi vẽ OpenCV
# - Tốc độ: thủ công (MANUAL_SPEED) hoặc auto theo lệch làn
# - Phím: m bật/tắt overlay mask; q thoát
# ==============================================================

import socket as pysock
import cv2
import numpy as np
import time
import json
import base64
from ultralytics import YOLO
import logging
import road  # cần road.py: centerline_offset_yellow & lane_offset

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"
PORT = 54321

RESIZE_W, RESIZE_H = 640, 480

# PID lái (độ)
KP = 0.06
KI = 0.0
KD = 0.03
ANGLE_LIMIT = 25
PIXEL_TO_DEG = 0.18  # quy đổi px -> độ

# ---- Tốc độ ----
ENABLE_MANUAL_SPEED = True   # True: dùng MANUAL_SPEED; False: auto theo lệch làn
MANUAL_SPEED = 50            # khi ENABLE_MANUAL_SPEED=True, chỉnh số này
MIN_SPEED = 30               # min khi auto; cũng là tốc độ tối thiểu khi cua gắt

# Làm mịn offset
SMOOTH_OFFSET = True
ALPHA_OFF = 0.2
DEADZONE_PX = 6

# ---- Stop/Go vật cản ----
OBST_CLASSES = {0, 1, 2, 3, 5, 7}   # person,bicycle,car,motorcycle,bus,truck
DETECT_CONF  = 0.60
CLEAR_CONF   = 0.30
STOP_HOLD_FRAMES  = 10
CLEAR_HOLD_FRAMES = 10
SLOW_CONF    = 0.45

# ROI phát hiện vật cản
ROI_Y_MIN_FRAC = 0.55
ROI_MIN_H_FRAC = 0.20

# ROI bám vàng
YELLOW_ROI_TOP    = 300
YELLOW_ROI_HEIGHT = 220
YELLOW_MIN_AREA   = 400
# ==============================================

# ---------- PID ----------
class PID:
    def __init__(self, kp, ki, kd, out_min=-ANGLE_LIMIT, out_max=ANGLE_LIMIT):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integral = 0.0; self.prev_error = 0.0; self.prev_time = None

    def reset(self):
        self.integral = 0.0; self.prev_error = 0.0; self.prev_time = None

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0
        dt = now - self.prev_time
        if dt <= 0: dt = 1e-3
        self.prev_time = now

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        out = max(self.out_min, min(self.out_max, out))
        return out

# ---------- Socket helpers ----------
def connect_map():
    s = pysock.socket(pysock.AF_INET, pysock.SOCK_STREAM)
    s.connect((HOST, PORT))
    print(f"[OK] Connected to {HOST}:{PORT}")
    return s

JSON_DECODER = json.JSONDecoder()
def iter_json_objects(sock, buf_holder, chunk_size=65536):
    chunk = sock.recv(chunk_size)
    if not chunk:
        return
    buf_holder['buf'] += chunk.decode('utf-8', errors='replace')
    s = buf_holder['buf']; i, n = 0, len(s)
    while i < n and s[i].isspace(): i += 1
    while i < n:
        try:
            obj, end = JSON_DECODER.raw_decode(s, i)
            yield obj
            i = end
            while i < n and s[i].isspace(): i += 1
        except json.JSONDecodeError:
            break
    buf_holder['buf'] = s[i:]

def send_control(sock, angle, speed):
    msg = f"{int(round(angle))} {int(speed)}"
    sock.sendall(msg.encode("utf-8"))

# ---------- Speed helper ----------
def adaptive_speed_control(error_px, base_speed=90, min_speed=MIN_SPEED):
    e = abs(error_px)
    speed = base_speed * (1 - (e / 280.0) ** 1.8)
    if e > 140: speed *= 0.8
    if e > 180: speed *= 0.7
    return int(max(speed, min_speed))

# ================== MAIN ==================
if __name__ == "__main__":
    sock = connect_map()
    pid_steer = PID(KP, KI, KD)
    yolo = YOLO("yolov8n.pt")

    cv2.namedWindow("AGV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AGV View", 900, 520)
    cv2.moveWindow("AGV View", 40, 80)

    cv2.namedWindow("Yellow Mask (BW)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yellow Mask (BW)", 900, 260)
    cv2.moveWindow("Yellow Mask (BW)", 40, 620)

    show_overlay = True
    print("[KEY] 'm' toggle overlay, 'q' quit")

    state = "DRIVING"
    stop_counter = 0
    clear_counter = 0
    off_smooth = None
    last_ts = time.time()

    buf_holder = {'buf': ''}
    running = True

    try:
        # "đánh thức" simulator nếu cần
        send_control(sock, 0, 0)

        while running:
            produced = False
            for pkg in iter_json_objects(sock, buf_holder):
                produced = True

                # ===== Nhận & giải mã ảnh =====
                try:
                    img_bytes = base64.b64decode(pkg["Img"])
                    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None: continue
                except Exception as e:
                    print("[ERR decode]", e); continue

                frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
                H, W = frame.shape[:2]
                cx = int(W // 2)

                # ===== 1) OFFSET theo vạch vàng (OpenCV) =====
                off, mask_vis = road.centerline_offset_yellow(
                    frame,
                    roi_top=YELLOW_ROI_TOP,
                    roi_height=YELLOW_ROI_HEIGHT,
                    min_area=YELLOW_MIN_AREA,
                    visualize=True
                )

                # Hiển thị mask
                if mask_vis is not None:
                    gray_mask = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2GRAY) if mask_vis.ndim == 3 else mask_vis
                    cv2.imshow("Yellow Mask (BW)", gray_mask)
                else:
                    cv2.imshow("Yellow Mask (BW)", np.zeros((RESIZE_H//3, RESIZE_W), dtype=np.uint8))

                # Fallback bám vạch trắng khi không thấy vàng
                if off is None:
                    off_white, _ = road.lane_offset(frame, roi_top=300, white=True)
                    if off_white is not None:
                        off = off_white

                # Làm mịn + deadzone
                if SMOOTH_OFFSET and off is not None:
                    if off_smooth is None:
                        off_smooth = float(off)
                    else:
                        off_smooth = (1 - ALPHA_OFF) * off_smooth + ALPHA_OFF * float(off)
                    off_use = off_smooth
                else:
                    off_use = off if off is not None else None
                    if off is None:
                        off_smooth = None

                if off_use is not None and abs(off_use) < DEADZONE_PX:
                    off_use = 0.0

                # ===== 2) PID lái =====
                if off_use is None:
                    steer_cmd = 0.0
                    pid_steer.reset()
                else:
                    error_deg = -float(off_use) * float(PIXEL_TO_DEG)
                    steer_cmd = pid_steer.update(error_deg)
                steer_cmd = max(-ANGLE_LIMIT, min(ANGLE_LIMIT, steer_cmd))

                # ===== 3) YOLO vật cản =====
                res = yolo(frame, imgsz=640, conf=0.25, classes=list(OBST_CLASSES), verbose=False)[0]
                best_conf = 0.0
                obstacle_present = False
                for b in res.boxes:
                    cls = int(b.cls)
                    if cls not in OBST_CLASSES: continue
                    c = float(b.conf)
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    h_box = y2 - y1
                    if y2 < int(H * ROI_Y_MIN_FRAC): continue
                    if h_box < int(H * ROI_MIN_H_FRAC): continue
                    best_conf = max(best_conf, c)
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255) if c >= DETECT_CONF else (0, 165, 255), 2)
                    if c >= DETECT_CONF: obstacle_present = True

                # ===== 4) State machine Stop/Go =====
                if obstacle_present:
                    stop_counter += 1; clear_counter = 0
                else:
                    clear_counter += 1; stop_counter = 0

                if state == "DRIVING" and stop_counter >= STOP_HOLD_FRAMES:
                    state = "STOPPED"
                elif state == "STOPPED" and clear_counter >= CLEAR_HOLD_FRAMES and best_conf <= CLEAR_CONF:
                    state = "DRIVING"

                # ===== 5) Tốc độ =====
                if ENABLE_MANUAL_SPEED:
                    speed_cmd = MANUAL_SPEED
                else:
                    # auto theo lệch làn, giảm tốc khi cua gắt
                    if off_use is None:
                        speed_cmd = MIN_SPEED
                    else:
                        speed_cmd = adaptive_speed_control(off_use, base_speed=90, min_speed=MIN_SPEED)

                if state != "DRIVING":
                    speed_cmd = 0
                else:
                    # Phanh nhẹ theo góc lái để an toàn khi cua
                    speed_cmd = max(MIN_SPEED, int(speed_cmd - abs(steer_cmd) * 1.2))

                # ===== 6) Gửi lệnh =====
                send_control(sock, steer_cmd, speed_cmd)

                # ===== 7) Overlay & HUD =====
                # Vẽ overlay mask vào vùng ROI
                if show_overlay and mask_vis is not None:
                    y0 = int(YELLOW_ROI_TOP)
                    h_roi, w_roi = mask_vis.shape[:2]
                    if y0 + h_roi <= frame.shape[0]:
                        roi_disp = frame[y0:y0 + h_roi, 0:w_roi]
                        if mask_vis.shape[:2] != roi_disp.shape[:2]:
                            mask_vis = cv2.resize(mask_vis, (roi_disp.shape[1], roi_disp.shape[0]))
                        if mask_vis.ndim == 2:
                            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                        blended = cv2.addWeighted(roi_disp, 0.6, mask_vis, 0.4, 0)
                        frame[y0:y0 + h_roi, 0:w_roi] = blended

                # Crosshair
                y_cross = int(min(YELLOW_ROI_TOP + YELLOW_ROI_HEIGHT//2, H-1))
                cv2.line(frame, (0, y_cross), (W, y_cross), (255,255,255), 2)
                cv2.line(frame, (cx, 0), (cx, H), (255,255,255), 1)
                cv2.circle(frame, (cx, y_cross), 6, (255,255,255), -1)

                # HUD
                off_show = 0 if off_use is None else int(off_use)
                cv2.putText(frame, f"Err: {off_show:+d}px", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Steer: {int(round(steer_cmd))}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"Speed: {int(speed_cmd)} ({state})", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                mode_text = "Manual" if ENABLE_MANUAL_SPEED else "Auto"
                cv2.putText(frame, f"Mode: {mode_text}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                cv2.imshow("AGV View", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
                elif key == ord('m'):
                    show_overlay = not show_overlay

            if not produced:
                # chưa parse được JSON hoàn chỉnh -> lặp tiếp để recv thêm
                pass

    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[DONE] disconnected")
