# client_test.py
# Kết nối simulator + bám tâm vàng (double yellow) + PID + Stop/Go vật cản + YOLO
# + Hiển thị song song: AGV View (camera) & Yellow Mask (BW)
# + Phím 'm' bật/tắt overlay mask lên camera, 'q' thoát

import socket
import json
import base64
import cv2
import numpy as np
import time

from ultralytics import YOLO
import road  # road.py phải có centerline_offset_yellow

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"
PORT = 54321
MAX_BUF = 700_000
RESIZE_W, RESIZE_H = 640, 480

# PID lái
KP = 0.06
KI = 0.0
KD = 0.03
ANGLE_LIMIT = 25
PIXEL_TO_DEG = 0.18

# Tốc độ
BASE_SPEED = 70
MIN_SPEED  = 25

# Làm mịn offset
SMOOTH_OFFSET = True
ALPHA_OFF = 0.2

# Deadzone px quanh tâm
DEADZONE_PX = 6

# ==== DỪNG/TRÁNH VẬT CẢN ====
OBST_CLASSES = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, motorcycle, bus, truck
DETECT_CONF  = 0.60
CLEAR_CONF   = 0.30
STOP_HOLD_S  = 0.6
CLEAR_HOLD_S = 0.6
SLOW_CONF    = 0.45

# ROI phát hiện vật cản
ROI_Y_MIN_FRAC = 0.55
ROI_MIN_H_FRAC = 0.20

# ROI bám vàng
YELLOW_ROI_TOP     = 300
YELLOW_ROI_HEIGHT  = 220
YELLOW_MIN_AREA    = 400
# ==============================================


class PID:
    def __init__(self, kp, ki, kd, out_min=-ANGLE_LIMIT, out_max=ANGLE_LIMIT):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.prev_time = now

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, out))


def connect_map():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.settimeout(0.1)
    print(f"[OK] Connected to map {HOST}:{PORT}")
    return s


def recv_frame(sock):
    try:
        data = sock.recv(MAX_BUF)
    except socket.timeout:
        return None, None, None
    except Exception as e:
        print("[ERR recv]", e)
        return None, None, None

    if not data:
        return None, None, None

    try:
        pkg = json.loads(data)
    except Exception as e:
        print("[ERR json]", e)
        return None, None, None

    try:
        img_bytes = base64.b64decode(pkg["Img"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("[ERR decode img]", e)
        return None, None, None

    return frame, pkg.get("Angle", 0.0), pkg.get("Speed", 0.0)


def send_control(sock, angle, speed):
    msg = f"{angle} {speed}"
    try:
        sock.sendall(msg.encode("utf-8"))
    except Exception as e:
        print("[ERR send]", e)


if __name__ == "__main__":
    sock = connect_map()
    pid_steer = PID(KP, KI, KD)
    model = YOLO("yolov8s.pt")

    # === CỬA SỔ ===
    cv2.namedWindow("AGV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AGV View", 900, 520)
    cv2.moveWindow("AGV View", 40, 80)

    cv2.namedWindow("Yellow Mask (BW)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yellow Mask (BW)", 900, 260)
    cv2.moveWindow("Yellow Mask (BW)", 40, 620)

    show_overlay = True
    print("[KEY] Press 'm' to toggle overlay, 'q' to quit.")

    state = "DRIVING"
    seen_timer = 0.0
    clear_timer = 0.0
    last_ts = time.time()
    resume_boost = 0.0
    off_smooth = None

    try:
        while True:
            send_control(sock, 0, 0)

            frame, sim_angle, sim_speed = recv_frame(sock)
            if frame is None:
                cv2.imshow("Yellow Mask (BW)", np.zeros((200, 200), dtype=np.uint8))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            H, W = frame.shape[:2]

            # ===== 1) OFFSET theo vạch vàng =====
            off, mask_vis = road.centerline_offset_yellow(
                frame,
                roi_top=YELLOW_ROI_TOP,
                roi_height=YELLOW_ROI_HEIGHT,
                min_area=YELLOW_MIN_AREA,
                visualize=True
            )

            # Hiển thị mask trắng-đen
            if mask_vis is not None:
                if len(mask_vis.shape) == 3:
                    gray_mask = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2GRAY)
                else:
                    gray_mask = mask_vis
                cv2.imshow("Yellow Mask (BW)", gray_mask)
            else:
                cv2.imshow("Yellow Mask (BW)", np.zeros((RESIZE_H//3, RESIZE_W), dtype=np.uint8))

            # Fallback bám vạch trắng
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
                error_deg = -off_use * PIXEL_TO_DEG
                steer_cmd = pid_steer.update(error_deg)

            # ===== 3) YOLO vật cản =====
            t0 = time.time()
            dt = t0 - last_ts
            last_ts = t0
            res = model(frame, imgsz=640, conf=0.25, verbose=False)[0]

            best_conf = 0.0
            obstacle_present = False
            for b in res.boxes:
                cls = int(b.cls)
                if cls not in OBST_CLASSES:
                    continue
                c = float(b.conf)
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                h_box = y2 - y1
                if y2 < int(H * ROI_Y_MIN_FRAC): continue
                if h_box < int(H * ROI_MIN_H_FRAC): continue
                best_conf = max(best_conf, c)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255) if c >= DETECT_CONF else (0, 165, 255), 2)
                if c >= DETECT_CONF: obstacle_present = True

            # ===== 4) STATE MACHINE =====
            if obstacle_present:
                seen_timer += dt
                clear_timer = 0.0
            else:
                clear_timer += dt
                seen_timer = 0.0

            if state == "DRIVING":
                if seen_timer >= STOP_HOLD_S:
                    state = "STOPPED"
                    resume_boost = 0.0
                elif best_conf >= SLOW_CONF:
                    speed_cmd = max(MIN_SPEED, int(BASE_SPEED * 0.5))
                else:
                    speed_cmd = BASE_SPEED
            else:
                speed_cmd = 0
                if clear_timer >= CLEAR_HOLD_S and best_conf <= CLEAR_CONF:
                    state = "DRIVING"
                    resume_boost = 0.0

            if state == "DRIVING":
                speed_cmd = max(MIN_SPEED, speed_cmd - abs(steer_cmd) * 1.2)
                resume_boost = min(resume_boost + dt * 20, 20)
                speed_cmd = min(speed_cmd + resume_boost, BASE_SPEED)

            steer_cmd = max(-25, min(25, steer_cmd))
            send_control(sock, steer_cmd, int(speed_cmd))

            # ===== 5) Overlay an toàn (đã fix lỗi kích thước/kênh) =====
            if show_overlay and mask_vis is not None:
                y0 = YELLOW_ROI_TOP
                h_roi, w_roi = mask_vis.shape[:2]
                if y0 + h_roi <= frame.shape[0]:
                    roi_disp = frame[y0:y0+h_roi, 0:w_roi]

                    # resize mask nếu lệch kích thước
                    if mask_vis.shape[:2] != roi_disp.shape[:2]:
                        mask_vis = cv2.resize(mask_vis, (roi_disp.shape[1], roi_disp.shape[0]))

                    # chuyển gray → BGR nếu cần
                    if len(mask_vis.shape) == 2:
                        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

                    blended = cv2.addWeighted(roi_disp, 0.6, mask_vis, 0.4, 0)
                    frame[y0:y0+h_roi, 0:w_roi] = blended

            # ===== 6) HUD =====
            cv2.putText(frame, f"offset: {0 if off_use is None else int(off_use)} px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"PID steer: {steer_cmd:.1f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            cv2.putText(frame, f"state: {state}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            cv2.imshow("AGV View", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_overlay = not show_overlay

    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("[DONE] disconnected")
