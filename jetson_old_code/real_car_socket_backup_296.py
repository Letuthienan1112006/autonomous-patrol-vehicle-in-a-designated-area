"""
real_car_socket.py
Bridge xe thật cho patrol_robot.py

Flow đúng với patrol_robot.py hiện tại:

    patrol_robot.py gửi command: "steer speed"
            ↓
    real_car_socket.py nhận command
            ↓
    gửi "steer speed\n" xuống STM32 qua /dev/ttyACM0
            ↓
    đọc frame từ camera C270
            ↓
    gửi JSON {"Img": "...base64..."} về patrol_robot.py

Cách chạy:
    Terminal 1: python3 real_car_socket.py
    Terminal 2: python3 patrol_robot.py
"""

import socket
import cv2
import base64
import json
import time
import glob
import threading

try:
    import serial
except ImportError:
    serial = None


# ============================================================
# CONFIG
# ============================================================
HOST = "127.0.0.1"
PORT = 54321

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
JPEG_QUALITY = 70
LOOP_DELAY = 0.03

STM32_BAUDRATE = 115200
STM32_PORT_CANDIDATES = [
    "/dev/ttyACM0",
    "/dev/ttyACM1",
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
]

STEER_MIN = -30
STEER_MAX = 30
SPEED_MIN = 0
SPEED_MAX = 25

ALLOW_DRY_RUN = True
WATCHDOG_TIMEOUT = 1.0


# ============================================================
# UTILS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def find_stm32_port():
    for port in STM32_PORT_CANDIDATES:
        if glob.glob(port):
            return port
    return None


def open_stm32():
    if serial is None:
        print("[STM32] Chưa có pyserial. Cài bằng: python3 -m pip install --user pyserial")
        return None

    port = find_stm32_port()
    if port is None:
        print("[STM32] Không thấy /dev/ttyACM* hoặc /dev/ttyUSB*")
        if ALLOW_DRY_RUN:
            print("[STM32] DRY RUN: không gửi motor, chỉ test camera/socket")
            return None
        raise RuntimeError("Không tìm thấy STM32")

    try:
        ser = serial.Serial(port, STM32_BAUDRATE, timeout=0.05, write_timeout=0.1)
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[STM32] Connected: {port} @ {STM32_BAUDRATE}")
        return ser
    except Exception as e:
        print(f"[STM32] Lỗi mở {port}: {e}")
        if ALLOW_DRY_RUN:
            print("[STM32] DRY RUN: không gửi motor, chỉ test camera/socket")
            return None
        raise


def send_to_stm32(ser, steer, speed):
    steer = clamp(int(float(steer)), STEER_MIN, STEER_MAX)
    speed = clamp(int(float(speed)), SPEED_MIN, SPEED_MAX)

    msg = f"{steer} {speed}\n"

    if ser is None:
        return

    try:
        ser.write(msg.encode("ascii"))
    except Exception as e:
        print(f"[STM32] Send error: {e}")


def stm32_reader(ser, stop_event):
    if ser is None:
        return

    while not stop_event.is_set():
        try:
            line = ser.readline()
            if line:
                msg = line.decode("ascii", errors="ignore").strip()
                if msg:
                    print(f"[STM32 RX] {msg}")
        except Exception:
            time.sleep(0.1)


# ============================================================
# CAMERA INIT
# ============================================================
print("[REAL CAR] Đang mở camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("[REAL CAR] LỖI: Không mở được camera!")
    print("[REAL CAR] Kiểm tra /dev/video0 hoặc đổi CAMERA_INDEX")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("[REAL CAR] LỖI: Camera mở được nhưng không đọc được frame!")
    exit(1)

print(f"[REAL CAR] Camera OK! Frame size: {frame.shape}")


# ============================================================
# STM32 INIT
# ============================================================
ser = open_stm32()
stop_event = threading.Event()

rx_thread = threading.Thread(
    target=stm32_reader,
    args=(ser, stop_event),
    daemon=True,
)
rx_thread.start()


# ============================================================
# SOCKET SERVER
# ============================================================
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"[REAL CAR] Đang lắng nghe tại {HOST}:{PORT}")
print("[REAL CAR] Chờ patrol_robot.py kết nối...")
print("[REAL CAR] Nhấn Ctrl+C để dừng")

conn, addr = server.accept()
print(f"[REAL CAR] Đã kết nối: {addr}")


# ============================================================
# MAIN LOOP
# ============================================================
frame_count = 0
start_time = time.time()
last_cmd = ""
last_cmd_time = time.time()
last_watchdog_stop = 0

try:
    while True:
        # 1. Nhận command từ patrol_robot.py
        try:
            cmd = conn.recv(1024).decode(errors="ignore").strip()
        except ConnectionResetError:
            print("[REAL CAR] patrol_robot.py đã ngắt kết nối")
            break

        if not cmd:
            continue

        parts = cmd.split()

        if len(parts) >= 2:
            try:
                steer = float(parts[0])
                speed = float(parts[1])
            except ValueError:
                steer, speed = 0, 0
        else:
            steer, speed = 0, 0

        # 2. Gửi xuống STM32
        send_to_stm32(ser, steer, speed)
        last_cmd_time = time.time()

        if cmd != last_cmd:
            print(f"[CMD] steer={steer:.1f} speed={speed:.1f}")
            last_cmd = cmd

        # 3. Watchdog an toàn
        if time.time() - last_cmd_time > WATCHDOG_TIMEOUT:
            if time.time() - last_watchdog_stop > 0.5:
                send_to_stm32(ser, 0, 0)
                print("[WATCHDOG] No command -> STOP")
                last_watchdog_stop = time.time()

        # 4. Đọc camera
        ret, frame = cap.read()
        if not ret:
            print("[REAL CAR] Mất frame, retry...")
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # 5. Encode JPEG
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            continue

        # 6. Gửi JSON thô về patrol_robot.py
        # KHÔNG dùng length-prefix vì patrol_robot.py đang json.loads(raw)
        payload = {"Img": base64.b64encode(buf).decode()}
        data = json.dumps(payload).encode()

        try:
            conn.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            print("[REAL CAR] Connection broken")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[REAL CAR] FPS: {fps:.1f} | Frames sent: {frame_count}")

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("\n[REAL CAR] Đang dừng...")

finally:
    print("[REAL CAR] STOP motor...")
    try:
        send_to_stm32(ser, 0, 0)
        time.sleep(0.1)
        send_to_stm32(ser, 0, 0)
    except Exception:
        pass

    stop_event.set()

    if ser is not None:
        ser.close()

    cap.release()

    try:
        conn.close()
    except Exception:
        pass

    server.close()
    print("[REAL CAR] Đã đóng camera/socket/serial")
