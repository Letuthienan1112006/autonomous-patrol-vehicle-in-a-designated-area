"""
Confg.py — Cấu hình tập trung cho hệ thống AGV Patrol Robot
============================================================
Tất cả các file Python khác sẽ import từ file này:
    from Confg import *

Khi cần đổi 1 thông số → CHỈ sửa ở đây, không sửa các file khác.

Tổ chức theo 7 nhóm:
  1. NETWORK    — socket xe + MQTT
  2. TELEGRAM   — bot token, chat id
  3. CAMERA     — kích thước, FPS
  4. MODEL      — đường dẫn AI models
  5. VISION     — ngưỡng detection
  6. CONTROL    — PID, tốc độ, steering
  7. NAVIGATION — patrol locations, ArUco, intersection
  8. ALERT      — cooldown, lưu ảnh
============================================================
"""

# ============================================================
#  1. NETWORK — Socket xe (Jetson ↔ simulator/STM32) + MQTT
# ============================================================

# Socket xe (Python ↔ STM32 hoặc fake_car_socket)
SOCKET_IP   = "127.0.0.1"
SOCKET_PORT = 54321

# MQTT broker (server.py chạy trên PC)
MQTT_BROKER_IP    = "192.168.1.100"   # IP máy chủ chạy Mosquitto
MQTT_PORT         = 1883
MQTT_TOPIC_ALERT  = "patrol/alert"     # khi phát hiện xâm nhập
MQTT_TOPIC_STATUS = "patrol/status"    # heartbeat mỗi 10 giây
MQTT_CLIENT_ID    = "jetson_patrol_01"


# ============================================================
#  2. TELEGRAM — Bot thông báo cảnh báo
# ============================================================

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"   # lấy từ @BotFather
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"     # ID nhóm bảo vệ (số âm)


# ============================================================
#  3. CAMERA
# ============================================================

CAMERA_INDEX  = 0       # 0 = C270 mặc định
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 360
CAMERA_FPS    = 30


# ============================================================
#  4. MODEL PATHS — đường dẫn các file model AI
# ============================================================

SEGMENTATION_MODEL_PATH = "/mnt/ssd/agv_models/segmentation.tar"
SIGN_MODEL_PATH         = "/mnt/big/detectsign.pt"
PERSON_MODEL_PATH       = "yolov8n.pt"   # Jetson compatible YOLOv8n

# Tham số model
SEG_INPUT_SIZE = (360, 640)              # (H, W) cho segmentation
SEG_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
SEG_NORMALIZE_STD  = [0.229, 0.224, 0.225]


# ============================================================
#  5. VISION — Ngưỡng detection
# ============================================================

# --- Person detection (YOLO) ---
PERSON_CONF_THRESHOLD = 0.55    # ngưỡng confidence (0-1)
PERSON_MIN_AREA       = 1500    # diện tích bbox tối thiểu (pixel²)
PERSON_CONFIRM_FRAMES = 8       # số frame liên tiếp để xác nhận
PERSON_CONFIRM_RATIO  = 0.75    # tỉ lệ frame phải detect được
PERSON_CLASS_ID       = 0       # class ID 0 = person trong COCO

# --- Segmentation (DeepLabV3) ---
SEG_NUM_CLASSES = 2
SEG_BACKBONE    = "resnet50"
CLASS_COLORS    = {0: [0, 0, 0], 1: [0, 255, 0]}  # 0=nền, 1=lane


# ============================================================
#  6. CONTROL — Điều khiển xe
# ============================================================

# --- Tốc độ ---
BASE_SPEED = 90    # tốc độ mặc định trên thẳng
MIN_SPEED  = 70    # tốc độ tối thiểu khi cua

# --- Steering ---
MAX_STEER  = 20    # góc lái tối đa (độ)
STEER_CENTER_X = 320   # tâm khung hình (640/2)

# --- PID lane following ---
PID_KP = 0.16
PID_KI = 0.001
PID_KD = 0.01
PID_MAX_OUTPUT  = 20
PID_INTEGRAL_LIMIT = 8      # anti-windup
PID_STEP_LIMIT  = 10        # max delta giữa 2 lần compute

# --- Adaptive speed ---
ADAPTIVE_ERROR_NORM      = 280     # chuẩn hóa error → factor
ADAPTIVE_POWER           = 1.8     # mũ giảm tốc theo error
ADAPTIVE_HEAVY_THRESHOLD = 140     # error > 140 → giảm tốc thêm 0.8x
ADAPTIVE_SEVERE_THRESHOLD = 180    # error > 180 → giảm tốc thêm 0.7x

# --- Slope detection ---
SLOPE_BOOST_LIGHT      = 1.10   # error > 50
SLOPE_BOOST_MEDIUM     = 1.15   # error > 100
SLOPE_BOOST_HEAVY      = 1.20   # khi lane points thay đổi nhiều


# ============================================================
#  7. NAVIGATION — Tuần tra, ArUco, giao lộ
# ============================================================

# --- Vị trí tuần tra ---
PATROL_LOCATIONS = [
    "Cong chinh",
    "Khu A - Hanh lang",
    "Khu B - Bai xe",
    "Khu C - San sau",
]
LOCATION_INTERVAL = 60   # giây — đổi vị trí mỗi 60s

# --- ArUco markers ---
ARUCO_DICT_NAME      = "DICT_4X4_50"     # tên dict trong cv2.aruco
ARUCO_MIN_AREA       = 800                # pixel² — bỏ marker quá nhỏ
ARUCO_CONFIRM_FRAMES = 5                  # số frame liên tiếp để nhận lệnh

# Marker generation (cho generate_markers.py)
ARUCO_MARKER_SIZE  = 600     # pixel — kích thước ảnh marker
ARUCO_BORDER_SIZE  = 80      # pixel — viền trắng xung quanh
ARUCO_OUTPUT_DIR   = "./markers"

# Ánh xạ marker ID → lệnh
ARUCO_COMMANDS = {
    0: "turn_left",    # marker ID 0 → rẽ trái
    1: "turn_right",   # marker ID 1 → rẽ phải
    2: "straight",     # marker ID 2 → đi thẳng
    3: "stop",         # marker ID 3 → dừng lại
}

# Hiển thị nhãn marker khi in ra
ARUCO_MARKER_INFO = {
    0: {"cmd": "RE TRAI",  "color": (220, 100, 50),  "symbol": "<-"},
    1: {"cmd": "RE PHAI",  "color": (50,  100, 220), "symbol": "->"},
    2: {"cmd": "DI THANG", "color": (50,  180, 50),  "symbol": "^"},
    3: {"cmd": "DUNG LAI", "color": (50,  50,  220), "symbol": "[]"},
}

# --- Logic dừng khi gặp marker STOP ---
STOP_DURATION = 3.0   # giây dừng khi gặp marker STOP

# --- Intersection detection ---
INTERSECTION_SCAN_Y           = 270   # đường scan để detect lane center
INTERSECTION_DISPLAY_SCAN_Y   = 360   # đường scan để vẽ display
INTERSECTION_SHIFT_THRESHOLD  = 80    # pixel — ngưỡng dịch chuyển detect rẽ
INTERSECTION_STABLE_FRAMES    = 3     # số frame ổn định để kích hoạt
INTERSECTION_MAX_FRAMES       = 25    # frame tối đa ở chế độ intersection
INTERSECTION_TIMEOUT_FRAMES   = 40    # frame timeout reset sign

# Khoảng cách an toàn sau khi đi qua marker
SAFE_DIST_FRAMES_AFTER_SIGN = 5

# Lane post-intersection stability
LANE_STABLE_OFFSET = 40   # |midpoint - center_x| < 40 → coi như đã thẳng
LANE_STABLE_WIDTH  = 70   # right - left > 70 → lane đủ rộng

# Stability filter (chống nhiễu giao lộ)
STABILITY_REQUIRED_FRAMES    = 3
STABILITY_CONFIDENCE_THR     = 0.6
STABILITY_AGREEMENT_RATIO    = 0.7


# ============================================================
#  8. ALERT & LOGGING
# ============================================================

ALERT_COOLDOWN_SEC = 30                     # giây chờ giữa 2 lần alert
SAVE_ALERT_IMAGE   = True
ALERT_IMAGE_DIR    = "./alert_images"
STATUS_INTERVAL_SEC = 10                    # gửi heartbeat mỗi 10s

# --- Server side (Sever.py) ---
ALERT_SERVER_SAVE_DIR = "./server_alerts"
PATROL_DB_PATH        = "./patrol_log.db"


# ============================================================
#  9. DEBUG
# ============================================================

SHOW_DEBUG_WINDOW = False    # Headless test qua SSH
DEBUG_HUD_FONT_SCALE = 0.52
DEBUG_HUD_LINE_HEIGHT = 21


# ============================================================
#  10. LIDAR (chưa dùng — để dành cho v2)
# ============================================================

# Đơn vị: mét
LIDAR_STOP_DIST_M           = 1.20
LIDAR_RESUME_DIST_M         = 1.60
LIDAR_MAX_VALID_DIST_M      = 12.0
LIDAR_STOP_CONFIRM_FRAMES   = 3
LIDAR_CLEAR_CONFIRM_FRAMES  = 8


# ============================================================
#  HELPER — Tạo các giá trị dẫn xuất (không sửa)
# ============================================================
import os
import cv2

# Lấy ArUco dict object từ tên string
def _get_aruco_dict():
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAME)
    )

# Đảm bảo các thư mục tồn tại
def _ensure_dirs():
    os.makedirs(ALERT_IMAGE_DIR, exist_ok=True)
    os.makedirs(ARUCO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ALERT_SERVER_SAVE_DIR, exist_ok=True)

# Auto-call khi import
_ensure_dirs()