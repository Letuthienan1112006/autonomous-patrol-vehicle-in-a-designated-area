"""
real_car_socket.py - Bridge xe that v3 cho patrol_robot.py.

Kien truc:
  - Main: nhan mot command AI, sau do gui dung mot frame JSON.
  - Sender: gui STM32 moi 100 ms, doc lap voi toc do AI.
  - LiDAR: doc scan va cap nhat LidarSafety.
  - STM32 reader: doc telemetry.

Thu tu uu tien an toan:
  shutdown/serial fault > LiDAR STOP > AI timeout > command AI.

Port co dinh:
  STM32: /dev/ttyACM0
  LiDAR: /dev/ttyUSB0

Protocol voi patrol_robot.py hien tai khong co length-prefix. Vi vay bridge
giu nghiem ngat mot command hop le -> mot frame JSON, khong gui frame tu do.
"""

import base64
import json
import math
import socket
import sys
import threading
import time

import cv2
from rplidar import RPLidar

try:
    import serial
except ImportError:
    serial = None

from lidar_safety import LidarSafety, SCAN_TIMEOUT_SEC


HOST = "127.0.0.1"
PORT = 54321

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CAMERA_FPS = 30
JPEG_QUALITY = 70

STM32_PORT = "/dev/ttyACM0"
STM32_BAUDRATE = 115200
STM32_READ_TIMEOUT = 0.05

LIDAR_PORT = "/dev/ttyUSB0"

STEER_MIN = -30
STEER_MAX = 30
SPEED_MIN = 0
SPEED_MAX = 25

SEND_INTERVAL = 0.10
AI_TIMEOUT = 1.0
SOCKET_POLL_TIMEOUT = 0.20

DRY_RUN = "--dry-run" in sys.argv


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class SharedCommand:
    def __init__(self):
        self._lock = threading.Lock()
        self._steer = 0.0
        self._speed = 0.0
        self._last_update = None

    def set(self, steer, speed):
        if not math.isfinite(steer) or not math.isfinite(speed):
            raise ValueError("command khong huu han")

        with self._lock:
            self._steer = steer
            self._speed = speed
            self._last_update = time.monotonic()

    def get(self):
        with self._lock:
            return self._steer, self._speed, self._last_update

    def invalidate(self):
        with self._lock:
            self._steer = 0.0
            self._speed = 0.0
            self._last_update = None


class LidarWorker:
    def __init__(self, safety, stop_event):
        self.safety = safety
        self.stop_event = stop_event
        self._lidar = None
        self._lidar_lock = threading.Lock()

    def run(self):
        lidar = None

        try:
            lidar = RPLidar(LIDAR_PORT, timeout=SCAN_TIMEOUT_SEC)
            with self._lidar_lock:
                self._lidar = lidar

            print(f"[LIDAR] Connected: {LIDAR_PORT}")
            print(f"[LIDAR] Info: {lidar.get_info()}")

            health = lidar.get_health()
            print(f"[LIDAR] Health: {health}")
            if not health or health[0] != "Good":
                raise RuntimeError(f"health khong Good: {health}")

            for scan in lidar.iter_scans():
                if self.stop_event.is_set():
                    break
                self.safety.update(scan)

        except Exception as error:
            if not self.stop_event.is_set():
                print(
                    f"[LIDAR] LOI: {type(error).__name__}: {error} - ep STOP",
                    flush=True,
                )
                self.safety.force_stop(
                    f"lidar loi: {type(error).__name__}: {error}"
                )
        finally:
            self._close_lidar(lidar)
            with self._lidar_lock:
                self._lidar = None
            print("[LIDAR] Thread da dung", flush=True)

    def shutdown(self):
        self.safety.force_stop("bridge shutdown")

        with self._lidar_lock:
            lidar = self._lidar

        # Dong ket noi tu main de danh thuc iter_scans() neu no dang block.
        self._close_lidar(lidar)

    @staticmethod
    def _close_lidar(lidar):
        if lidar is None:
            return

        try:
            lidar.stop()
        except Exception:
            pass
        try:
            lidar.stop_motor()
        except Exception:
            pass
        try:
            lidar.disconnect()
        except Exception:
            pass


def open_camera():
    print("[CAMERA] Dang mo camera...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not camera.isOpened():
        raise RuntimeError(f"khong mo duoc camera index={CAMERA_INDEX}")

    ok, frame = camera.read()
    if not ok:
        camera.release()
        raise RuntimeError("camera mo duoc nhung khong doc duoc frame")

    print(f"[CAMERA] OK: frame={frame.shape}")
    return camera


def open_stm32():
    if DRY_RUN:
        print("[STM32] DRY RUN: khong mo UART, khong the dieu khien motor")
        return None

    if serial is None:
        raise RuntimeError("chua cai pyserial")

    try:
        stm32 = serial.Serial(
            STM32_PORT,
            STM32_BAUDRATE,
            timeout=STM32_READ_TIMEOUT,
            write_timeout=0.1,
        )
        time.sleep(2.0)
        stm32.reset_input_buffer()
        stm32.reset_output_buffer()
        print(f"[STM32] Connected: {STM32_PORT} @ {STM32_BAUDRATE}")
        return stm32
    except Exception:
        raise


def write_stm32(stm32, steer, speed):
    if stm32 is None:
        return True

    final_steer = clamp(int(float(steer)), STEER_MIN, STEER_MAX)
    final_speed = clamp(int(float(speed)), SPEED_MIN, SPEED_MAX)
    message = f"{final_steer} {final_speed}\n"

    try:
        stm32.write(message.encode("ascii"))
        return True
    except Exception as error:
        print(f"[STM32] LOI GHI: {type(error).__name__}: {error}", flush=True)
        return False


def sender_thread_fn(stm32, ai_command, safety, stop_event):
    last_status = None

    try:
        while not stop_event.is_set():
            loop_started = time.monotonic()

            steer, speed, last_update = ai_command.get()
            reasons = []

            if last_update is None:
                reasons.append("chua nhan lenh AI")
            elif time.monotonic() - last_update > AI_TIMEOUT:
                reasons.append("AI timeout")

            safety.check_timeout()
            snapshot = safety.snapshot()
            if snapshot["state"] == "STOP":
                reasons.append(f"LIDAR STOP ({snapshot['reason']})")

            if reasons:
                final_steer, final_speed = 0, 0
                status = "STOP: " + "; ".join(reasons)
            else:
                final_steer, final_speed = steer, speed
                status = "AI ALLOWED"

            if status != last_status:
                print(f"[SENDER] {status}", flush=True)
                last_status = status

            if not write_stm32(stm32, final_steer, final_speed):
                stop_event.set()
                break

            elapsed = time.monotonic() - loop_started
            stop_event.wait(max(0.0, SEND_INTERVAL - elapsed))

    except Exception as error:
        print(f"[SENDER] LOI: {type(error).__name__}: {error}", flush=True)
        stop_event.set()
    finally:
        # Sender la noi duy nhat ghi UART, tranh hai thread ghi chen nhau.
        for _ in range(5):
            write_stm32(stm32, 0, 0)
            time.sleep(0.02)
        print("[SENDER] Da gui STOP cuoi cung", flush=True)


def stm32_reader_fn(stm32, stop_event):
    if stm32 is None:
        return

    while not stop_event.is_set():
        try:
            raw_line = stm32.readline()
            if raw_line:
                message = raw_line.decode("ascii", errors="ignore").strip()
                if message:
                    print(f"[STM32 RX] {message}", flush=True)
        except Exception as error:
            if not stop_event.is_set():
                print(
                    f"[STM32] LOI DOC: {type(error).__name__}: {error}",
                    flush=True,
                )
                stop_event.set()
            break


def parse_ai_command(raw_data):
    message = raw_data.decode("utf-8", errors="ignore").strip()
    parts = message.split()

    if len(parts) < 2:
        raise ValueError(f"command ngan: {message!r}")

    steer = float(parts[0])
    speed = float(parts[1])
    if not math.isfinite(steer) or not math.isfinite(speed):
        raise ValueError(f"command khong huu han: {message!r}")

    return steer, speed, message


def encode_frame(camera, max_retries=3, retry_delay=0.02):
    ok = False
    frame = None

    for _attempt in range(max_retries):
        ok, frame = camera.read()
        if ok:
            break
        time.sleep(retry_delay)

    if not ok:
        raise RuntimeError(
            f"mat frame camera (da thu lai {max_retries} lan)"
        )

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG encode fail")

    payload = {"Img": base64.b64encode(buffer.tobytes()).decode("ascii")}
    return json.dumps(payload).encode("utf-8")


def run_socket_server(camera, ai_command, safety, stop_event):
    server = None
    connection = None

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(SOCKET_POLL_TIMEOUT)

        print(f"[NET] Lang nghe tai {HOST}:{PORT}")
        print("[NET] Cho patrol_robot.py ket noi...")

        while connection is None and not stop_event.is_set():
            try:
                connection, address = server.accept()
                print(f"[NET] Da ket noi: {address}")
            except socket.timeout:
                continue

        if connection is None:
            return

        connection.settimeout(SOCKET_POLL_TIMEOUT)

        frame_count = 0
        started = time.monotonic()
        last_printed_command = None

        while not stop_event.is_set():
            try:
                raw_command = connection.recv(1024)
            except socket.timeout:
                # Khong gui frame khi khong co command, tranh lech protocol.
                continue
            except (ConnectionResetError, OSError) as error:
                print(f"[NET] Mat ket noi: {error}")
                break

            if raw_command == b"":
                print("[NET] patrol_robot.py da dong ket noi")
                break

            try:
                steer, speed, command_text = parse_ai_command(raw_command)
            except ValueError as error:
                print(f"[NET] Lenh khong hop le: {error}")
                ai_command.invalidate()
                continue

            ai_command.set(steer, speed)
            if command_text != last_printed_command:
                print(f"[CMD] steer={steer:.1f} speed={speed:.1f}")
                last_printed_command = command_text

            try:
                frame_data = encode_frame(camera)
                connection.sendall(frame_data)
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError) as error:
                print(f"[NET] Gui frame loi: {error}")
                break

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                snapshot = safety.snapshot()
                print(
                    f"[NET] FPS={frame_count / elapsed:.1f} "
                    f"frames={frame_count} LIDAR={snapshot['state']} "
                    f"reason={snapshot['reason']}"
                )

    finally:
        ai_command.invalidate()
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


def main():
    camera = None
    stm32 = None
    sender_thread = None
    reader_thread = None
    lidar_thread = None

    stop_event = threading.Event()
    ai_command = SharedCommand()
    safety = LidarSafety()
    lidar_worker = LidarWorker(safety, stop_event)

    try:
        camera = open_camera()
        stm32 = open_stm32()

        sender_thread = threading.Thread(
            target=sender_thread_fn,
            args=(stm32, ai_command, safety, stop_event),
            daemon=True,
        )
        reader_thread = threading.Thread(
            target=stm32_reader_fn,
            args=(stm32, stop_event),
            daemon=True,
        )
        lidar_thread = threading.Thread(
            target=lidar_worker.run,
            daemon=True,
        )

        sender_thread.start()
        reader_thread.start()
        lidar_thread.start()

        run_socket_server(camera, ai_command, safety, stop_event)

    except KeyboardInterrupt:
        print("\n[BRIDGE] Ctrl+C - E-STOP", flush=True)
    except Exception as error:
        print(f"[BRIDGE] LOI: {type(error).__name__}: {error}", flush=True)
    finally:
        print("[BRIDGE] Dang shutdown...", flush=True)
        ai_command.invalidate()
        safety.force_stop("bridge shutdown")
        stop_event.set()

        # Dong LiDAR de danh thuc iter_scans() neu dang block.
        lidar_worker.shutdown()

        if sender_thread is not None:
            sender_thread.join(timeout=2.0)
        if lidar_thread is not None:
            lidar_thread.join(timeout=2.0)
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)

        if stm32 is not None:
            try:
                stm32.close()
            except Exception:
                pass
        if camera is not None:
            camera.release()

        print("[BRIDGE] Da dong camera/socket/serial/LiDAR", flush=True)


if __name__ == "__main__":
    main()
