import cv2
import numpy as np
import time

# ============================================================
# 1️⃣  BÁM VẠCH TRẮNG / ĐEN
# ============================================================
def lane_offset(frame, roi_top=300, white=True, th_white=200, th_black=80):
    """
    Tính độ lệch so với vạch trắng hoặc đen.
    Trả về (offset_px, mask_binary)
    """
    roi = frame[roi_top:, :]
    L = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)[:, :, 1]  # lấy kênh sáng (L)

    if white:
        _, m = cv2.threshold(L, th_white, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(L, th_black, 255, cv2.THRESH_BINARY_INV)

    # Lọc nhiễu
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)

    M = cv2.moments(m)
    off = None
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = m.shape[1] // 2
        off = cx - center
        cv2.circle(roi, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(roi, (center, 0), (center, roi.shape[0] - 1), (255, 0, 0), 1)
    return off, m


# ============================================================
# 2️⃣  BÁM VẠCH VÀNG (DOUBLE-YELLOW)
# ============================================================
def centerline_offset_yellow(frame, roi_top=280, roi_height=180, min_area=800, visualize=True):
    """
    Bám tâm vùng vàng giữa hai vạch vàng.
    Trả về:
      - offset_px: cx - (W//2), >0 lệch phải, <0 lệch trái
      - mask_vis: ảnh BGR hoặc gray để hiển thị
    """
    H, W = frame.shape[:2]
    y0 = max(0, roi_top)
    y1 = min(H, roi_top + roi_height)
    if y1 <= y0:
        return None, None

    roi = frame[y0:y1, 0:W]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Dải màu vàng (điều chỉnh tuỳ ánh sáng)
    lower1 = np.array([15, 80, 120], np.uint8)
    upper1 = np.array([35, 255, 255], np.uint8)
    lower2 = np.array([20, 50, 90], np.uint8)
    upper2 = np.array([40, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Lọc nhiễu
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contour lớn nhất
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if visualize else None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if visualize else None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if visualize else None

    cx = int(M["m10"] / M["m00"])
    offset_px = cx - (W // 2)

    # Tạo mask hiển thị
    if visualize:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cy = int((y1 - y0) * 0.6)
        cv2.circle(mask_vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(mask_vis, (W // 2, 0), (W // 2, y1 - y0), (255, 0, 0), 2)
    else:
        mask_vis = None

    return offset_px, mask_vis
