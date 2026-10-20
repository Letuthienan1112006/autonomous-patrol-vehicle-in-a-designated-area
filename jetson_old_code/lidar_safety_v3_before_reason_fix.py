"""
lidar_safety.py - Lop an toan LiDAR cho AGV.

- FRONT_CENTER_DEG=130.0 da duoc xac nhan bang log thuc te.
- Hysteresis STOP/GO dung cac nguong trong Confg.py.
- Scan loi, scan qua it diem hoac timeout deu ep STOP.
- Tat ca state deu duoc bao ve bang RLock de dung an toan giua cac thread.
"""

import math
import threading
import time

from rplidar import RPLidar

from Confg import (
    LIDAR_CLEAR_CONFIRM_FRAMES,
    LIDAR_MAX_VALID_DIST_M,
    LIDAR_RESUME_DIST_M,
    LIDAR_STOP_CONFIRM_FRAMES,
    LIDAR_STOP_DIST_M,
)


PORT = "/dev/ttyUSB0"

FRONT_CENTER_DEG = 130.0
FRONT_CONE_DEG = 30.0

MIN_QUALITY = 5
MIN_VALID_DIST_MM = 100.0
MAX_VALID_DIST_MM = LIDAR_MAX_VALID_DIST_M * 1000.0
MIN_VALID_SCAN_POINTS = 20
SCAN_TIMEOUT_SEC = 1.0

STOP_DIST_MM = LIDAR_STOP_DIST_M * 1000.0
RESUME_DIST_MM = LIDAR_RESUME_DIST_M * 1000.0


def normalize_angle(angle):
    """Dua goc ve mien [-180, 180]."""
    result = angle % 360.0
    if result > 180.0:
        result -= 360.0
    return result


def analyze_scan(scan):
    """
    Tra ve (scan_valid, closest_point, valid_point_count).

    closest_point la (distance_mm, relative_angle_deg, quality), hoac None
    neu scan hop le nhung khong co diem phan xa trong cone phia truoc.
    """
    valid_point_count = 0
    front_points = []

    for quality, raw_angle, distance in scan:
        if quality < MIN_QUALITY:
            continue
        if not math.isfinite(distance):
            continue
        if not (MIN_VALID_DIST_MM <= distance <= MAX_VALID_DIST_MM):
            continue

        valid_point_count += 1
        relative_angle = normalize_angle(raw_angle - FRONT_CENTER_DEG)

        if -FRONT_CONE_DEG <= relative_angle <= FRONT_CONE_DEG:
            front_points.append((distance, relative_angle, quality))

    if valid_point_count < MIN_VALID_SCAN_POINTS:
        return False, None, valid_point_count

    if not front_points:
        return True, None, valid_point_count

    return True, min(front_points, key=lambda point: point[0]), valid_point_count


def closest_front_point(scan):
    """Tra diem hop le gan nhat trong cone phia truoc, hoac None."""
    scan_valid, point, _ = analyze_scan(scan)
    if not scan_valid:
        return None
    return point


class LidarSafety:
    def __init__(self):
        self._lock = threading.RLock()

        self.state = "STOP"
        self.reason = "startup"
        self.stop_confirm = 0
        self.clear_confirm = 0

        self.last_valid_scan_time = None
        self.last_dist_mm = None
        self.last_angle = None
        self.last_valid_point_count = 0
        self.last_scan_valid = False

    def force_stop(self, reason):
        with self._lock:
            changed = self.state != "STOP" or self.reason != reason

            self.state = "STOP"
            self.reason = reason
            self.stop_confirm = LIDAR_STOP_CONFIRM_FRAMES
            self.clear_confirm = 0

            if changed:
                print(f"[LIDAR-SAFETY] FORCE STOP ({reason})", flush=True)

            return self.state

    def update(self, scan):
        scan_valid, point, valid_count = analyze_scan(scan)

        with self._lock:
            self.last_scan_valid = scan_valid
            self.last_valid_point_count = valid_count

            if not scan_valid:
                self.last_dist_mm = None
                self.last_angle = None
                return self.force_stop(
                    f"scan khong hop le: {valid_count} diem"
                )

            self.last_valid_scan_time = time.monotonic()

            if point is None:
                distance = None
                angle = None
            else:
                distance, angle, _quality = point

            self.last_dist_mm = distance
            self.last_angle = angle

            obstacle_close = (
                distance is not None and distance <= STOP_DIST_MM
            )
            path_clear = distance is None or distance >= RESUME_DIST_MM

            if obstacle_close:
                self.stop_confirm += 1
                self.clear_confirm = 0
            elif path_clear:
                self.clear_confirm += 1
                self.stop_confirm = 0
            else:
                # Dai hysteresis: giu nguyen trang thai hien tai.
                self.stop_confirm = 0
                self.clear_confirm = 0

            if self.state == "GO":
                if self.stop_confirm >= LIDAR_STOP_CONFIRM_FRAMES:
                    self.state = "STOP"
                    self.reason = "obstacle"
                    print(
                        f"[LIDAR-SAFETY] -> STOP (dist={distance:.0f}mm)",
                        flush=True,
                    )
            elif self.clear_confirm >= LIDAR_CLEAR_CONFIRM_FRAMES:
                self.state = "GO"
                self.reason = "clear"
                print("[LIDAR-SAFETY] -> GO", flush=True)

            return self.state

    def check_timeout(self):
        with self._lock:
            if self.last_valid_scan_time is None:
                return self.force_stop("chua co scan hop le")

            elapsed = time.monotonic() - self.last_valid_scan_time
            if elapsed > SCAN_TIMEOUT_SEC:
                return self.force_stop(
                    f"mat scan qua {SCAN_TIMEOUT_SEC:.1f}s"
                )

            return self.state

    def is_stop(self):
        with self._lock:
            return self.state == "STOP"

    def snapshot(self):
        """Doc mot snapshot atomic de thread khac khong thay state nua cu nua moi."""
        with self._lock:
            return {
                "state": self.state,
                "reason": self.reason,
                "dist_mm": self.last_dist_mm,
                "angle": self.last_angle,
                "valid_points": self.last_valid_point_count,
                "scan_valid": self.last_scan_valid,
            }


def main():
    safety = LidarSafety()
    lidar = None

    try:
        lidar = RPLidar(PORT, timeout=SCAN_TIMEOUT_SEC)

        print("=== LIDAR INFO ===")
        print(lidar.get_info())

        print("=== HEALTH ===")
        health = lidar.get_health()
        print(health)
        if not health or health[0] != "Good":
            raise RuntimeError(f"LiDAR health khong Good: {health}")

        print("=== SAFETY TEST (HYSTERESIS) ===")
        print(
            f"STOP <= {STOP_DIST_MM:.0f}mm | "
            f"RESUME >= {RESUME_DIST_MM:.0f}mm | "
            f"FRONT_CENTER_DEG={FRONT_CENTER_DEG:.1f}"
        )
        print("Dua hop carton truoc xe. Ctrl+C de dung.")

        for scan in lidar.iter_scans():
            state = safety.update(scan)
            snapshot = safety.snapshot()

            if not snapshot["scan_valid"]:
                print(
                    f"[LIDAR] STOP  INVALID SCAN  "
                    f"points={snapshot['valid_points']}"
                )
            elif snapshot["dist_mm"] is None:
                print(
                    f"[LIDAR] {state:4s}  no front obstacle  "
                    f"points={snapshot['valid_points']}"
                )
            else:
                print(
                    f"[LIDAR] {state:4s}  "
                    f"dist={snapshot['dist_mm']:5.0f} mm  "
                    f"angle={snapshot['angle']:+6.1f} deg  "
                    f"points={snapshot['valid_points']}"
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        safety.force_stop("Ctrl+C")
        print("\nStopping...")
    except Exception as error:
        safety.force_stop(f"{type(error).__name__}: {error}")
        print(f"\n[LIDAR-SAFETY] LOI: {type(error).__name__}: {error}")
    finally:
        if lidar is not None:
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except Exception as error:
                print(f"[LIDAR-SAFETY] Loi khi dong ket noi: {error}")

        print("DONE - safety state = STOP")


if __name__ == "__main__":
    main()
