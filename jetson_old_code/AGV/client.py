# ==============================================================
# client.py  (fixed)
# Bám làn giữa (DeepLabV3) + Stop/Go khi gặp vật cản (YOLOv8)
# - Sửa lỗi OpenCV "Can't parse 'center'" bằng cách ép int thuần
# - Fix JSON "Extra data" với buffer + raw_decode
# - Gửi lệnh 1 lần/khung
# ==============================================================

import socket as pysock
import cv2
import numpy as np
import time
import json
import base64
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import logging

# gọn log YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ================== CẤU HÌNH ==================
HOST = "127.0.0.1"
PORT = 54321

RESIZE_W, RESIZE_H = 640, 360   # kích thước dùng cho segmentation/hiển thị
SCAN_Y = 270                    # hàng y để lấy midpoint trên mask (theo RESIZE_H)

# PID lái
KP = 0.16
KI = 0.001
KD = 0.01
ANGLE_LIMIT = 20  # độ

# Tốc độ
BASE_SPEED = 60
MIN_SPEED  = 40

# Stop/Go
STOP_HOLD_FRAMES  = 10
CLEAR_HOLD_FRAMES = 10

# YOLO các lớp phải dừng (COCO)
STOP_CLASSES = {0, 1, 2, 3, 5, 7}   # person,bicycle,car,motorcycle,bus,truck
STOP_CONF   = 0.55
CLEAR_CONF  = 0.35
ROI_Y_MIN_FRAC = 0.45   # chỉ tính box có đáy y2 > 45% ảnh (gần xe)
ROI_MIN_H_FRAC = 0.18   # và chiều cao box đủ lớn

YOLO_WEIGHTS = "yolov8n.pt"  # nhẹ, nhanh; đổi "yolov8s.pt" nếu cần chính xác hơn

# ================== MODEL ==================
from model import create_deeplabv3  # file model.py đã chỉnh theo weights enum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model = create_deeplabv3(num_classes=2, backbone="resnet50", pretrained=False)
seg_model = seg_model.to(device)
seg_model.eval()

yolo = YOLO(YOLO_WEIGHTS)

# ================== TIỀN XỬ LÝ ==================
img_transform = transforms.Compose([
    transforms.Resize((RESIZE_H, RESIZE_W)),  # 360 x 640
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ================== HÀM PHỤ ==================
class PID:
    def __init__(self, kp, ki, kd, out_min=-ANGLE_LIMIT, out_max=ANGLE_LIMIT):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integral = 0.0; self.prev_error = 0.0
        self.prev_time = None
        self.last_output = 0.0

    def reset(self):
        self.integral = 0.0; self.prev_error = 0.0; self.prev_time = None
        self.last_output = 0.0

    def update(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0
        dt = now - self.prev_time
        self.prev_time = now
        if dt < 0.001: dt = 0.001

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -8.0, 8.0))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp*error + self.ki*self.integral + self.kd*derivative
        # rate limit để đánh lái mượt
        delta = float(np.clip(out - self.last_output, -10.0, 10.0))
        out = self.last_output + delta

        out = float(np.clip(out, self.out_min, self.out_max))
        self.last_output = out
        return out

def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """mask: HxW (0: nền, 1: lane) -> ảnh màu BGR để overlay."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[mask == 1] = (0, 255, 0)
    return color

def get_lane_midpoint(mask: np.ndarray, y: int = SCAN_Y):
    """Tìm (left, right, mid) tại hàng y. Trả về int thuần hoặc None."""
    if y >= mask.shape[0]:
        y = mask.shape[0] - 1
    row = mask[int(y), :]
    idx = np.where(row == 1)[0]
    if len(idx) < 2:
        return None
    left = int(idx[0])
    right = int(idx[-1])
    mid = int((left + right) // 2)
    return left, right, mid

def adaptive_speed_control(error_px: int, base_speed=BASE_SPEED, min_speed=MIN_SPEED) -> int:
    e = abs(error_px)
    speed_factor = 1 - (e / 280.0) ** 1.8
    if e > 140: speed_factor *= 0.8
    if e > 180: speed_factor *= 0.7
    return int(max(base_speed * speed_factor, min_speed))

# --- Fix JSON dính gói ---
JSON_DECODER = json.JSONDecoder()
def iter_json_objects(sock, buf_holder, chunk_size=65536):
    chunk = sock.recv(chunk_size)
    if not chunk:
        return
    buf_holder['buf'] += chunk.decode('utf-8', errors='replace')

    s = buf_holder['buf']; i = 0; n = len(s)
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

# ================== MAIN ==================
if __name__ == "__main__":
    print("[INFO] Start: center-lane follow + stop/go obstacle (no traffic signs)")
    sock = pysock.socket(pysock.AF_INET, pysock.SOCK_STREAM)
    sock.connect((HOST, PORT))
    # gói "đánh thức" nếu simulator cần
    sock.sendall(b"0 0")

    lane_pid = PID(KP, KI, KD)
    buf_holder = {'buf': ''}

    state = "DRIVING"
    stop_counter = 0
    clear_counter = 0
    last_mid = None

    # UI
    cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Debug", 900, 520)
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 900, 520)

    try:
        with torch.no_grad():
            running = True
            while running:
                produced = False
                for pkg in iter_json_objects(sock, buf_holder):
                    produced = True

                    # ===== Nhận & giải mã ảnh =====
                    try:
                        img_bytes = base64.b64decode(pkg["Img"])
                        arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                        if frame is None:
                            continue
                    except Exception as e:
                        print("[WARN] decode frame:", e)
                        continue

                    frame_disp = cv2.resize(frame, (RESIZE_W, RESIZE_H))
                    H, W = frame_disp.shape[:2]
                    cx = int(W // 2)
                    y_scan = int(min(SCAN_Y, H - 1))

                    # ===== Segmentation =====
                    pil_img = Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                    tensor = img_transform(pil_img).unsqueeze(0).to(device)
                    pred = seg_model(tensor)  # [1, C, H, W]
                    mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    lane_pts = get_lane_midpoint(mask, y=y_scan)
                    if lane_pts:
                        l, r, mid = lane_pts
                        # đảm bảo int thuần
                        l = int(l); r = int(r); mid = int(mid)
                        lane_err_px = int(mid - cx)
                        last_mid = mid
                    else:
                        lane_err_px = int(0 if last_mid is None else (int(last_mid) - cx))

                    # ===== PID + tốc độ =====
                    steer_deg = lane_pid.update(lane_err_px)
                    speed_cmd = adaptive_speed_control(lane_err_px)

                    # ===== YOLO vật cản =====
                    det_view = frame.copy()
                    H_full = det_view.shape[0]
                    results = yolo(det_view, conf=0.25, classes=list(STOP_CLASSES), verbose=False)
                    obstacle = False
                    best_conf = 0.0
                    for r in results:
                        if r.boxes is None:
                            continue
                        for b in r.boxes:
                            cls = int(b.cls[0]); conf = float(b.conf[0])
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            h_box = int(y2 - y1)
                            if y2 < int(H_full * ROI_Y_MIN_FRAC):  # chưa đủ gần
                                continue
                            if h_box < int(H_full * ROI_MIN_H_FRAC):  # box quá nhỏ
                                continue
                            best_conf = max(best_conf, conf)
                            color = (0,0,255) if conf >= STOP_CONF else (0,165,255)
                            cv2.rectangle(det_view, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            if conf >= STOP_CONF:
                                obstacle = True

                    # ===== State machine Stop/Go =====
                    if obstacle:
                        stop_counter += 1; clear_counter = 0
                    else:
                        clear_counter += 1; stop_counter = 0

                    if state == "DRIVING" and stop_counter >= STOP_HOLD_FRAMES:
                        state = "STOPPED"
                    elif state == "STOPPED" and clear_counter >= CLEAR_HOLD_FRAMES and best_conf <= CLEAR_CONF:
                        state = "DRIVING"

                    car_speed = 0 if state == "STOPPED" else speed_cmd

                    # ===== Gửi lệnh (1 lần/khung) =====
                    ctrl = f"{int(round(steer_deg))} {int(car_speed)}"
                    sock.sendall(ctrl.encode("utf-8"))

                    # ===== Hiển thị =====
                    color_mask = decode_segmap(mask)
                    debug = cv2.addWeighted(frame_disp, 0.55, color_mask, 0.45, 0.0)

                    # vẽ scan line + trục giữa (ÉP INT TUYỆT ĐỐI)
                    try:
                        cv2.line(debug, (0, int(y_scan)), (int(W), int(y_scan)), (255,255,0), 2)
                        cv2.line(debug, (int(cx), 0), (int(cx), int(H)), (180,180,180), 1)
                    except Exception as e:
                        print("[WARN draw line]", e)

                    if lane_pts:
                        try:
                            cv2.circle(debug, (int(mid), int(y_scan)), 6, (255,255,255), -1)
                            cv2.circle(debug, (int(l),   int(y_scan)), 4, (255,  0,  0), -1)
                            cv2.circle(debug, (int(r),   int(y_scan)), 4, (255,  0,  0), -1)
                            cv2.line(debug, (int(cx), int(y_scan)), (int(mid), int(y_scan)), (255,255,255), 2)
                        except Exception as e:
                            print("[WARN draw points]", type(mid), type(y_scan), e)

                    cv2.putText(debug, f"Err: {int(lane_err_px):+d}px", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv2.putText(debug, f"Steer: {int(round(steer_deg))}", (10, 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.putText(debug, f"Speed: {int(car_speed)} ({state})", (10, 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    cv2.imshow("Lane Debug", debug)
                    cv2.imshow("Detection", det_view)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        running = False
                        break

                if not produced:
                    # chưa parse được JSON hoàn chỉnh → vòng sau recv thêm
                    pass

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    except Exception as e:
        print(f"[ERR] {e}")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[DONE] Disconnected")
