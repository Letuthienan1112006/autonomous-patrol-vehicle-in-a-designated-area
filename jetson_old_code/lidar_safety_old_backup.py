from rplidar import RPLidar
import time

PORT = "/dev/ttyUSB0"

DANGER_DIST_MM = 500
SLOW_DIST_MM = 1000
FRONT_CONE_DEG = 30


def normalize_angle(angle):
    a = angle % 360
    if a > 180:
        a -= 360
    return a


def decide_from_scan(scan):
    front_points = []

    for quality, angle, distance in scan:
        if quality < 5:
            continue

        if distance < 100:
            continue

        a = normalize_angle(angle)

        if -FRONT_CONE_DEG <= a <= FRONT_CONE_DEG:
            front_points.append((distance, a, quality))

    if not front_points:
        return "CLEAR", None, None

    min_dist, min_angle, quality = min(front_points, key=lambda x: x[0])

    if min_dist < DANGER_DIST_MM:
        return "STOP", min_dist, min_angle

    if min_dist < SLOW_DIST_MM:
        return "SLOW", min_dist, min_angle

    return "CLEAR", min_dist, min_angle


lidar = RPLidar(PORT)

try:
    print("=== LIDAR INFO ===")
    print(lidar.get_info())

    print("=== HEALTH ===")
    print(lidar.get_health())

    print("=== SAFETY TEST ===")
    print("Đưa tay/vật cản trước LiDAR để xem STOP/SLOW/CLEAR")
    print("Ctrl+C để dừng")

    for scan in lidar.iter_scans():
        decision, dist, angle = decide_from_scan(scan)

        if dist is None:
            print("[LIDAR] CLEAR   no front obstacle")
        else:
            print(f"[LIDAR] {decision:5s}   dist={dist:5.0f} mm   angle={angle:+6.1f} deg")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    print("DONE")
