from rplidar import RPLidar
import time

PORT = '/dev/ttyUSB0'   # đổi nếu port khác
lidar = RPLidar(PORT)

print("=== INFO ===")
print(lidar.get_info())
print("Health:", lidar.get_health())

print("\n=== SCAN (5 vòng) ===")
try:
    for i, scan in enumerate(lidar.iter_scans()):
        print(f"\nScan {i+1}: {len(scan)} điểm")
        for q, angle, dist in scan[:5]:
            print(f"  góc={angle:6.1f}°  xa={dist:6.0f}mm  chất lượng={q}")
        if i >= 4:
            break
finally:
    lidar.stop()
    lidar.disconnect()
    print("\n✅ DONE")
