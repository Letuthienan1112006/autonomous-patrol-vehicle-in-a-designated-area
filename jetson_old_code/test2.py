import socket
import cv2
import numpy as np
import time
import json
import base64
import torch
from model import create_deeplabv3
from torchvision import transforms
from PIL import Image
from collections import deque
from ultralytics import YOLO
import logging
TRAFFIC_SIGN_CONF = 0.70   # biển báo
VEHICLE_CONF      = 0.70   # xe/người
VEHICLE_IOU       = 0.50   #gop bot khung trung
MIN_BOX_AREA_SIGN = 500
MIN_BOX_AREA_VEH  = 900

max_speeds = 60  # Tốc độ tối đa
min_speeds = 40  # Tốc độ tối thiểu

time_stop = 10.0  # thời gian dừng khi gặp xe khác

# Tắt logging của YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [0, 255, 0],
}

# YOLOv11 class names
YOLO_CLASSES = [
    "no_straight",  # 0 - cấm đi thẳng
    "no_turn_left",  # 1 - cấm rẽ trái
    "no_turn_right",  # 2 - cấm rẽ phải
    "stop",  # 3 - dừng lại
    "straight",  # 4 - chỉ được đi thẳng
    "turn_left",  # 5 - chỉ được rẽ trái
    "turn_right",  # 6 - chỉ được rẽ phải
    "unknown",  # 7 - không xác định
]
#YOLO_CLASSES += ["vehicle", "pedestrian", "bicycle"]  # Các lớp bổ sung nếu có


def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def get_lane_midpoint(mask, y=270):
    if y >= mask.shape[0]:
        y = mask.shape[0] - 1
    row = mask[y, :]
    lane_indices = np.where(row == 1)[0]

    if len(lane_indices) < 2:
        return None

    left = lane_indices[0]
    right = lane_indices[-1]

    mid = (left + right) // 2
    return left, right, mid


def detect_intersection_points(mask, scan_y=270):
    """
    Phát hiện các điểm giao lộ bằng cách phân tích nhiều điểm trên ảnh
    Trả về thông tin về các làn đường ở các vị trí khác nhau
    scan_y: Vị trí y để quét (mặc định 270)
    """
    height, width = mask.shape
    intersection_info = {
        "center_left": None,
        "center": None,
        "center_right": None,
        "left": None,
        "right": None,
        "is_intersection": False,
        "lane_count": 0,
        "should_turn_left": False,
        "should_turn_right": False,
        "left_shift": 0,
        "right_shift": 0,
        "confidence": 0.0,  # Thêm độ tin cậy
        "stability": 0.0,  # Thêm độ ổn định
    }

    # Sử dụng scan_y từ tham số thay vì cố định 270
    scan_y = min(scan_y, height - 1)

    # Đầu tiên, tìm ranh giới của toàn bộ vùng làn đường
    row = mask[scan_y, :]
    lane_pixels = np.where(row == 1)[0]

    if len(lane_pixels) < 10:
        return intersection_info

    # Ranh giới ngoài cùng của làn đường
    lane_leftmost = lane_pixels[0]
    lane_rightmost = lane_pixels[-1]
    lane_total_width = lane_rightmost - lane_leftmost

    # Định nghĩa các điểm kiểm tra với điểm trái/phải ở ngoài cùng
    scan_points = {
        "center": width // 2,
        "center_left": lane_leftmost + lane_total_width // 4,
        "center_right": lane_leftmost + 3 * lane_total_width // 4,
        "left": lane_leftmost,  # Điểm trái ở ngoài cùng bên trái
        "right": lane_rightmost,  # Điểm phải ở ngoài cùng bên phải
    }

    # Lưu vị trí ban đầu của điểm trái và phải để so sánh
    if not hasattr(detect_intersection_points, "initial_left"):
        detect_intersection_points.initial_left = lane_leftmost
        detect_intersection_points.initial_right = lane_rightmost
        detect_intersection_points.initial_center = width // 2

    # Phân tích từng điểm
    for point_name, x_pos in scan_points.items():
        if 0 <= x_pos < width:
            if point_name in ["left", "right"]:
                # Đối với điểm trái/phải, kiểm tra chính xác tại vị trí đó
                if mask[scan_y, x_pos] == 1:
                    intersection_info[point_name] = {
                        "detected": True,
                        "start": x_pos,
                        "end": x_pos,
                        "center": x_pos,
                        "width": 1,
                    }
                else:
                    intersection_info[point_name] = {"detected": False}
            else:
                # Đối với các điểm khác, kiểm tra vùng xung quanh
                x_start = max(0, x_pos - 30)
                x_end = min(width, x_pos + 30)

                row_segment = mask[scan_y, x_start:x_end]
                lane_pixels_segment = np.where(row_segment == 1)[0]

                if len(lane_pixels_segment) > 10:
                    lane_start = lane_pixels_segment[0] + x_start
                    lane_end = lane_pixels_segment[-1] + x_start
                    lane_center = (lane_start + lane_end) // 2

                    intersection_info[point_name] = {
                        "detected": True,
                        "start": lane_start,
                        "end": lane_end,
                        "center": lane_center,
                        "width": lane_end - lane_start,
                    }
                else:
                    intersection_info[point_name] = {"detected": False}
        else:
            intersection_info[point_name] = {"detected": False}

    # Đếm số làn phát hiện được trước khi sử dụng
    detected_lanes = sum(
        1
        for info in intersection_info.values()
        if isinstance(info, dict) and info.get("detected", False)
    )

    # Đảm bảo detected_lanes có giá trị hợp lệ
    detected_lanes = max(0, detected_lanes)

    # Tính toán độ dịch chuyển của điểm trái và phải
    if intersection_info["left"].get("detected"):
        intersection_info["left_shift"] = (
            intersection_info["left"]["center"]
            - detect_intersection_points.initial_left
        )

    if intersection_info["right"].get("detected"):
        intersection_info["right_shift"] = (
            intersection_info["right"]["center"]
            - detect_intersection_points.initial_right
        )

    # Ngưỡng động dựa trên tốc độ và điều kiện đường
    base_turn_threshold = 80
    # Điều chỉnh ngưỡng dựa trên số làn đường và độ rộng
    if detected_lanes >= 4:
        turn_threshold = base_turn_threshold * 1.3  # Tăng ngưỡng khi có nhiều làn
    elif detected_lanes == 3:
        turn_threshold = base_turn_threshold * 1.1
    else:
        turn_threshold = base_turn_threshold * 0.9  # Giảm ngưỡng khi ít làn

    # Tính độ tin cậy dựa trên độ nhất quán của shift
    left_shift_abs = abs(intersection_info["left_shift"])
    right_shift_abs = abs(intersection_info["right_shift"])

    # Độ tin cậy tăng khi shift lớn và số làn nhiều
    confidence_score = 0.0
    if detected_lanes >= 3:
        confidence_score += 0.3
    if max(left_shift_abs, right_shift_abs) > turn_threshold:
        confidence_score += 0.4
    if (
        detected_lanes >= 4
        and max(left_shift_abs, right_shift_abs) > turn_threshold * 1.2
    ):
        confidence_score += 0.3

    intersection_info["confidence"] = min(confidence_score, 1.0)

    # Phát hiện khi cần rẽ với điều kiện nghiêm ngặt hơn
    intersection_info["should_turn_left"] = (
        intersection_info["left_shift"] > turn_threshold
        and detected_lanes >= 3  # Tăng yêu cầu số làn
        and intersection_info["confidence"] > 0.6  # Yêu cầu độ tin cậy cao
    )
    intersection_info["should_turn_right"] = (
        intersection_info["right_shift"] < -turn_threshold
        and detected_lanes >= 3  # Tăng yêu cầu số làn
        and intersection_info["confidence"] > 0.6  # Yêu cầu độ tin cậy cao
    )

    intersection_info["lane_count"] = detected_lanes
    intersection_info["is_intersection"] = detected_lanes >= 3

    return intersection_info


def calculate_intersection_steering(intersection_info, traffic_sign_command):
    """
    Tính toán góc lái khi ở giao lộ dựa trên biển báo và độ dịch chuyển của các điểm
    """
    if not intersection_info["is_intersection"]:
        return None

    center_x = 320  # Trung tâm ảnh (width=640)

    # Ưu tiên điều khiển theo độ dịch chuyển của các điểm nếu không có biển báo
    if not traffic_sign_command:
        if intersection_info["should_turn_left"]:
            return -15  # Rẽ trái
        elif intersection_info["should_turn_right"]:
            return 15  # Rẽ phải
        else:
            # Giữ làn trung tâm
            center_lane = intersection_info.get("center")
            if center_lane and center_lane.get("detected"):
                current_center = center_lane["center"]
                steering_angle = (current_center - center_x) * 0.3
                return max(-5, min(5, steering_angle))
            return 0

    # Lấy thông tin làn trung tâm
    center_lane = intersection_info.get("center")
    if not center_lane or not center_lane.get("detected"):
        return None

    current_center = center_lane["center"]

    # Tính toán góc lái dựa trên lệnh biển báo - GIẢM GÓC LÁI ĐỂ TRÁNH VỤT QUA
    if traffic_sign_command == "turn_left":
        # Rẽ trái: sử dụng điểm left hoặc center_left
        if intersection_info["left"] and intersection_info["left"].get("detected"):
            target_x = intersection_info["left"]["center"]
        else:
            target_x = intersection_info.get("center_left", {}).get(
                "center", center_x - 80  # Giảm từ 100 xuống 80
            )

        steering_angle = (target_x - center_x) * 0.4  # Giảm từ 0.6 xuống 0.4
        return max(
            -15, min(-5, steering_angle)
        )  # Giảm góc lái: từ (-20,-8) xuống (-15,-5)

    elif traffic_sign_command == "turn_right":
        # Rẽ phải: sử dụng điểm right hoặc center_right
        if intersection_info["right"] and intersection_info["right"].get("detected"):
            target_x = intersection_info["right"]["center"]
        else:
            target_x = intersection_info.get("center_right", {}).get(
                "center", center_x + 80  # Giảm từ 100 xuống 80
            )

        steering_angle = (target_x - center_x) * 0.4  # Giảm từ 0.6 xuống 0.4
        return max(5, min(15, steering_angle))  # Giảm góc lái: từ (8,20) xuống (5,15)

    elif traffic_sign_command == "straight":
        # Đi thẳng: BỎ QUA hoàn toàn các signal rẽ, CHỈ tập trung giữ làn trung tâm
        # Nếu không có làn trung tâm rõ ràng, sử dụng làn hiện tại
        if center_lane and center_lane.get("detected"):
            steering_angle = (
                current_center - center_x
            ) * 0.25  # Giảm hệ số để ổn định hơn
            return max(-8, min(8, steering_angle))  # Cho phép điều chỉnh nhẹ hơn
        else:
            # Nếu không phát hiện được làn trung tâm, giữ hướng hiện tại
            return 0

    return None


def detect_lane_departure(intersection_info):
    """
    Phát hiện khi xe cần rẽ dựa trên sự thay đổi vị trí các điểm trái/phải
    """
    left_shift = intersection_info.get("left_shift", 0)
    right_shift = intersection_info.get("right_shift", 0)
    lane_count = intersection_info.get("lane_count", 0)

    # Ngưỡng để xác định cần rẽ - GIẢM để xe rẽ sớm hơn
    turn_threshold = 100  # Giảm từ 180 xuống 100

    # Yêu cầu cả độ dịch chuyển lớn VÀ có đủ làn đường (thực sự ở giao lộ)
    if left_shift > turn_threshold and lane_count >= 2:  # Giảm từ 3 xuống 2
        return "turn_left"
    elif right_shift < -turn_threshold and lane_count >= 2:  # Giảm từ 3 xuống 2
        return "turn_right"

    return None


img_transform = transforms.Compose(
    [
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_deeplabv3(num_classes=2, backbone="resnet50", pretrained=False)
model = model.to(device)

checkpoint_path = "/mnt/big/segmentation.tar"
checkpoint = torch.load(checkpoint_path, map_location=device)
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.eval()

# Load YOLOv11 model
yolo_model = YOLO("/mnt/big/detectsign.pt")
yolo_obj = YOLO("/mnt/big/yolov8n.pt")
COCO_NAMES=yolo_obj.names
VEHICLE_SET = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

steer_angle = 0
car_speed = 0

# Biến lưu trạng thái giao lộ và biển báo - sử dụng biến cờ thay vì thời gian
intersection_mode = False
traffic_sign_command = None

# Biến cờ để theo dõi trạng thái biển báo
sign_detected = False  # Đã phát hiện biển báo
passed_sign = False  # Đã đi qua biển báo (không còn thấy biển báo)
ready_to_turn = False  # Sẵn sàng rẽ (đã qua biển báo và đến giao lộ)
frames_without_sign = 0  # Đếm số frame không thấy biển báo để xác định đã qua


# Thêm bộ lọc ổn định cho quyết định rẽ - ĐIỀU CHỈNH CHO 60FPS
class StabilityFilter:
    def __init__(
        self, required_frames=3, confidence_threshold=0.6
    ):  # GIẢM từ 5->3 frames, 0.7->0.6 threshold
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        self.turn_left_buffer = deque(maxlen=required_frames)
        self.turn_right_buffer = deque(maxlen=required_frames)
        self.confidence_buffer = deque(maxlen=required_frames)

    def update(self, should_turn_left, should_turn_right, confidence):
        self.turn_left_buffer.append(should_turn_left)
        self.turn_right_buffer.append(should_turn_right)
        self.confidence_buffer.append(confidence)

    def get_stable_decision(self):
        if len(self.turn_left_buffer) < self.required_frames:
            return False, False, 0.0

        # Kiểm tra tính nhất quán trong buffer
        left_count = sum(self.turn_left_buffer)
        right_count = sum(self.turn_right_buffer)
        avg_confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)

        # Yêu cầu ít nhất 70% frame đồng ý và độ tin cậy cao - GIẢM NGƯỎNG CHO 60FPS
        stable_left = (
            left_count >= self.required_frames * 0.7  # GIẢM từ 0.8 xuống 0.7
            and avg_confidence >= self.confidence_threshold
        )
        stable_right = (
            right_count >= self.required_frames * 0.7  # GIẢM từ 0.8 xuống 0.7
            and avg_confidence >= self.confidence_threshold
        )

        return stable_left, stable_right, avg_confidence

    def reset(self):
        self.turn_left_buffer.clear()
        self.turn_right_buffer.clear()
        self.confidence_buffer.clear()


# Thêm bộ lọc ổn định cho việc phát hiện biển báo - ĐIỀU CHỈNH CHO 60FPS
class TrafficSignStabilityFilter:
    def __init__(
        self, required_frames=4, disappear_threshold=8
    ):  # GIẢM từ 8->4 và 15->8
        self.required_frames = (
            required_frames  # Số frame cần để xác nhận phát hiện biển báo
        )
        self.disappear_threshold = disappear_threshold  # Số frame liên tiếp mất tín hiệu để xác nhận đã qua biển báo
        self.sign_detection_buffer = deque(
            maxlen=required_frames
        )  # Buffer lưu tín hiệu phát hiện
        self.current_sign_buffer = deque(
            maxlen=required_frames
        )  # Buffer lưu loại biển báo
        self.consecutive_missing = 0  # Đếm số frame liên tiếp không thấy biển báo

    def update(self, sign_detected, current_sign=None):
        """
        Cập nhật trạng thái phát hiện biển báo
        sign_detected: bool - có phát hiện biển báo trong frame này không
        current_sign: str - loại biển báo phát hiện được (nếu có)
        """
        self.sign_detection_buffer.append(sign_detected)
        self.current_sign_buffer.append(current_sign if sign_detected else None)

        if sign_detected:
            self.consecutive_missing = 0
        else:
            self.consecutive_missing += 1

    def is_sign_stably_detected(self):
        """
        Kiểm tra biển báo có được phát hiện ổn định không
        Trả về: (bool, str) - (có phát hiện ổn định, loại biển báo)
        """
        if len(self.sign_detection_buffer) < self.required_frames:
            return False, None

        # Đếm số frame phát hiện biển báo trong buffer
        detection_count = sum(self.sign_detection_buffer)

        # Yêu cầu ít nhất 65% frame phát hiện được biển báo - GIẢM NGƯỎNG CHO 60FPS
        if detection_count >= self.required_frames * 0.65:  # GIẢM từ 0.75 xuống 0.65
            # Tìm biển báo xuất hiện nhiều nhất
            sign_counts = {}
            for sign in self.current_sign_buffer:
                if sign is not None:
                    sign_counts[sign] = sign_counts.get(sign, 0) + 1

            if sign_counts:
                most_common_sign = max(sign_counts, key=sign_counts.get)
                # Yêu cầu biển báo xuất hiện ít nhất 50% trong các frame có phát hiện - GIẢM NGƯỎNG
                if (
                    sign_counts[most_common_sign] >= detection_count * 0.5
                ):  # GIẢM từ 0.6 xuống 0.5
                    return True, most_common_sign

        return False, None

    def has_passed_sign(self):
        """
        Kiểm tra đã đi qua biển báo chưa (mất tín hiệu đủ lâu)
        """
        return self.consecutive_missing >= self.disappear_threshold

    def reset(self):
        """Reset tất cả buffer"""
        self.sign_detection_buffer.clear()
        self.current_sign_buffer.clear()
        self.consecutive_missing = 0


# THÊM: FPS Adapter để tự động điều chỉnh tham số cho 30-60FPS
class FPSAdapter:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.current_fps = target_fps
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_check = time.time()

    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()

        # Cập nhật FPS mỗi giây
        if current_time - self.last_fps_check >= 1.0:
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.current_fps = self.frame_count / elapsed
            self.last_fps_check = current_time

    def get_adapted_frames(self, base_frames_30fps):
        """Chuyển đổi số frame từ 30FPS sang FPS hiện tại"""
        fps_ratio = self.current_fps / 30.0
        return max(1, int(base_frames_30fps * fps_ratio))

    def get_adapted_threshold(self, base_threshold_30fps):
        """Điều chỉnh ngưỡng phần trăm cho FPS cao"""
        if self.current_fps > 50:  # Cho 60FPS
            return base_threshold_30fps * 0.8  # Giảm 20% ngưỡng
        elif self.current_fps > 40:  # Cho 45FPS
            return base_threshold_30fps * 0.9  # Giảm 10% ngưỡng
        return base_threshold_30fps  # Giữ nguyên cho 30FPS


# Khởi tạo FPS adapter
fps_adapter = FPSAdapter()


# Khởi tạo bộ lọc ổn định với FPS adapter - ĐIỀU CHỈNH DYNAMIC CHO 30-60FPS
def create_adaptive_filters():
    # Các tham số base cho 30FPS
    base_stability_frames = 3
    base_sign_frames = 4
    base_disappear_frames = 6

    # Điều chỉnh dựa trên FPS hiện tại
    adapted_stability_frames = fps_adapter.get_adapted_frames(base_stability_frames)
    adapted_sign_frames = fps_adapter.get_adapted_frames(base_sign_frames)
    adapted_disappear_frames = fps_adapter.get_adapted_frames(base_disappear_frames)

    # Điều chỉnh confidence threshold
    adapted_confidence = fps_adapter.get_adapted_threshold(0.6)
    adapted_sign_threshold = fps_adapter.get_adapted_threshold(0.65)

    stability_filter = StabilityFilter(
        required_frames=adapted_stability_frames,
        confidence_threshold=adapted_confidence,
    )

    traffic_sign_filter = TrafficSignStabilityFilter(
        required_frames=adapted_sign_frames,
        disappear_threshold=adapted_disappear_frames,
    )

    return stability_filter, traffic_sign_filter


# Khởi tạo filters với adaptive parameters
stability_filter, traffic_sign_filter = create_adaptive_filters()


class PIDController:
    def __init__(self, kp, ki, kd, max_output=20):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.error_history = deque(maxlen=5)
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()
        self.last_output = 0

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt < 0.001:
            dt = 0.001

        p_term = error * self.kp

        self.integral += error * dt
        if abs(self.integral) > 8:
            self.integral = 8 * np.sign(self.integral)
        i_term = self.integral * self.ki

        if len(self.error_history) > 0:
            avg_error = (error + self.last_error) / 2
            derivative = (avg_error - self.last_error) / dt
        else:
            derivative = (error - self.last_error) / dt

        self.last_error = error
        d_term = derivative * self.kd

        output = p_term + i_term + d_term

        max_change = 10
        if abs(output - self.last_output) > max_change:
            output = self.last_output + max_change * np.sign(output - self.last_output)

        if abs(output) > self.max_output:
            output = np.sign(output) * self.max_output

        self.error_history.append(error)
        self.last_output = output

        return int(output)


def adaptive_speed_control(error, base_speed=max_speeds, min_speed=min_speeds):
    """
    Điều khiển tốc độ adaptive - TĂNG TỐC ĐỘ ĐỂ LEO DỐC TỐT HƠN
    base_speed: 80->95 (+15)
    min_speed: 60->75 (+15)
    """
    error_abs = abs(error)
    speed_factor = 1 - (error_abs / 280) ** 1.8

    # Nới lỏng penalty khi lệch làn để duy trì tốc độ cao hơn
    if error_abs > 140:  # Tăng ngưỡng từ 120->140
        speed_factor *= 0.8  # Giảm penalty từ 0.75->0.8
    if error_abs > 180:  # Tăng ngưỡng từ 160->180
        speed_factor *= 0.7  # Giảm penalty từ 0.6->0.7

    speed = max(base_speed * speed_factor, min_speed)
    return int(speed)


def detect_steep_slope(lane_error, lane_points, prev_lane_points=None):
    """
    Phát hiện dốc cao dựa trên độ thay đổi vị trí làn đường
    Trả về hệ số tăng tốc (1.0 = bình thường, 1.2 = tăng 20%)
    """
    base_boost = 1.0

    # Nếu lệch làn lớn (có thể do dốc) -> tăng tốc để duy trì momentum
    if abs(lane_error) > 100:
        base_boost = 1.15  # Tăng 15%
    elif abs(lane_error) > 50:
        base_boost = 1.1  # Tăng 10%

    # Phát hiện thay đổi đột ngột vị trí làn (có thể do dốc)
    if lane_points and prev_lane_points:
        try:
            prev_left, prev_right, prev_mid = prev_lane_points
            curr_left, curr_right, curr_mid = lane_points

            # Tính độ thay đổi vị trí làn
            mid_change = abs(curr_mid - prev_mid)
            width_change = abs((curr_right - curr_left) - (prev_right - prev_left))

            # Nếu có thay đổi lớn -> có thể là dốc
            if mid_change > 30 or width_change > 20:
                base_boost = max(base_boost, 1.2)  # Tăng tối đa 20%

        except (TypeError, ValueError):
            pass  # Ignore errors in unpacking

    return base_boost


if __name__ == "__main__":
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.connect(("127.0.0.1", 54321))

    last_midpoint = None
    prev_lane_points = None  # THÊM: Lưu lane points trước đó để phát hiện dốc
    lane_pid = PIDController(kp=0.16, ki=0.001, kd=0.01, max_output=20)

    with torch.no_grad():
        try:
            while True:
                message = bytes(f"{steer_angle} {car_speed}", "utf-8")
                socket.sendall(message)
                data = socket.recv(100000)

                try:
                    recv_data = json.loads(data)

                    jpg_data = base64.b64decode(recv_data["Img"])
                    jpg_array = np.frombuffer(jpg_data, dtype=np.uint8)
                    cam_image = cv2.imdecode(jpg_array, flags=1)
                    pil_image = Image.fromarray(
                        cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
                    )
                    img_tensor = img_transform(pil_image).unsqueeze(0).to(device)

                    # DeepLabV3 inference
                    output = model(img_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                    # YOLOv11 inference để phát hiện biển báo
                    yolo_results = yolo_model(cam_image, conf=0.7)
                    detections = []
                    current_traffic_signs = []

                    for result in yolo_results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                bbox = box.xyxy[0].cpu().numpy()
                                width = bbox[2] - bbox[0]
                                height = bbox[3] - bbox[1]
                                box_area = width * height

                                min_box_area = 500
                                if box_area >= min_box_area:
                                    class_id = int(box.cls[0])
                                    detection = {
                                        "class": class_id,
                                        "confidence": float(box.conf[0]),
                                        "bbox": bbox,
                                        "area": box_area,
                                    }
                                    detections.append(detection)
                                    current_traffic_signs.append(YOLO_CLASSES[class_id])
                    #✅ Phát hiện xe/người bằng model COCO
                    veh_results = yolo_obj(
                        cam_image, conf=VEHICLE_CONF, iou=VEHICLE_IOU, classes=[0,1,2,3,5,7]  # person,bicycle,car,motorcycle,bus,truck
                    )
                    vehicle_boxes = []
                    for r in veh_results:
                        if r.boxes is None:
                            continue
                        for b in r.boxes:
                            cls_id = int(b.cls[0])
                            label = COCO_NAMES[cls_id]
                            if label in VEHICLE_SET:
                                x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                                conf = float(b.conf[0])
                                vehicle_boxes.append((x1, y1, x2, y2, label, conf))


                                    

                    # Phát hiện giao lộ và các điểm làn đường
                    # Đường cho PID (giữ nguyên ở y=270)
                    intersection_info = detect_intersection_points(pred_mask)

                    # Thêm bộ đếm ổn định cho giao lộ (THỰC HIỆN SAU KHI CÓ THÔNG TIN GIAO LỘ)
                    if not hasattr(
                        detect_intersection_points, "intersection_stable_count"
                    ):
                        detect_intersection_points.intersection_stable_count = 0

                    # Đếm số frame liên tiếp phát hiện giao lộ ổn định (ĐIỀU CHỈNH CHO 60FPS)
                    if (
                        intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3  # Giữ nguyên 3
                    ):
                        detect_intersection_points.intersection_stable_count += 1
                    else:
                        detect_intersection_points.intersection_stable_count = 0

                    # GIẢM MẠNH yêu cầu ổn định từ 6 xuống 3 frame cho 60FPS
                    intersection_truly_stable = (
                        detect_intersection_points.intersection_stable_count >= 3
                    )

                    lane_points = get_lane_midpoint(pred_mask, y=270)

                    # Cập nhật bộ lọc ổn định
                    stability_filter.update(
                        intersection_info.get("should_turn_left", False),
                        intersection_info.get("should_turn_right", False),
                        intersection_info.get("confidence", 0.0),
                    )

                    # Lấy quyết định ổn định từ bộ lọc
                    stable_left, stable_right, avg_confidence = (
                        stability_filter.get_stable_decision()
                    )

                    # THÊM: Điểm kiểm tra trước để phát hiện sớm khi đường thẳng sắp hết (y=200)
                    early_check_points = get_lane_midpoint(pred_mask, y=200)

                    # Đường cho hiển thị 5 điểm màu sắc (ở vị trí y=360)
                    intersection_info_display = detect_intersection_points(
                        pred_mask, scan_y=360
                    )

                    lane_error = 0

                    # Chỉ phát hiện lane departure khi có biển báo lệnh
                    suggested_turn = None

                    # Xử lý biển báo và cập nhật trạng thái bằng biến cờ
                    current_sign_found = False
                    has_command_sign = False  # Biến để kiểm tra có biển báo lệnh không
                    has_prohibition_sign = (
                        False  # Biến để kiểm tra có biển báo cấm không
                    )

                    # Xác định biển báo hiện tại để cập nhật vào bộ lọc
                    detected_sign = None
                    if current_traffic_signs:
                        for sign in current_traffic_signs:
                            # Xử lý biển báo LỆNH (cho phép)
                            if sign in ["turn_left", "turn_right", "straight"]:
                                has_command_sign = True
                                detected_sign = sign
                                current_sign_found = True
                                break

                            # Xử lý biển báo CẤM (chuyển đổi thành lệnh rẽ)
                            elif sign in [
                                "no_straight",
                                "no_turn_left",
                                "no_turn_right",
                            ]:
                                has_prohibition_sign = True
                                # Chuyển đổi biển cấm thành lệnh rẽ phù hợp
                                converted_command = None

                                if sign == "no_straight":
                                    # Cấm đi thẳng → kiểm tra trực tiếp sự hiện diện của làn đường
                                    left_lane_available = False
                                    right_lane_available = False

                                    # Kiểm tra làn bên trái
                                    if intersection_info.get(
                                        "left"
                                    ) and intersection_info["left"].get("detected"):
                                        left_lane_available = True
                                    elif intersection_info.get(
                                        "center_left"
                                    ) and intersection_info["center_left"].get(
                                        "detected"
                                    ):
                                        left_lane_available = True

                                    # Kiểm tra làn bên phải
                                    if intersection_info.get(
                                        "right"
                                    ) and intersection_info["right"].get("detected"):
                                        right_lane_available = True
                                    elif intersection_info.get(
                                        "center_right"
                                    ) and intersection_info["center_right"].get(
                                        "detected"
                                    ):
                                        right_lane_available = True

                                    # Quyết định hướng rẽ dựa trên làn đường có sẵn
                                    if left_lane_available and right_lane_available:
                                        # Cả hai hướng đều có đường → ưu tiên rẽ trái
                                        converted_command = "turn_left"
                                    elif left_lane_available:
                                        # Chỉ có đường bên trái
                                        converted_command = "turn_left"
                                    elif right_lane_available:
                                        # Chỉ có đường bên phải
                                        converted_command = "turn_right"
                                    else:
                                        # Không có đường rõ ràng → mặc định rẽ phải
                                        converted_command = "turn_right"

                                elif sign == "no_turn_left":
                                    # Cấm rẽ trái → KIỂM TRA ĐIỂM TRƯỚC để quyết định
                                    # Nếu điểm trước (y=200) vẫn có đường thẳng → đi thẳng
                                    # Nếu điểm trước không có đường thẳng → rẽ phải

                                    can_go_straight_ahead = (
                                        early_check_points is not None
                                    )

                                    if can_go_straight_ahead:
                                        # Vẫn còn đường phía trước → ưu tiên đi thẳng
                                        converted_command = "straight"
                                    else:
                                        # Không còn đường phía trước → phải rẽ
                                        right_lane_available = False

                                        if intersection_info.get(
                                            "right"
                                        ) and intersection_info["right"].get(
                                            "detected"
                                        ):
                                            right_lane_available = True
                                        elif intersection_info.get(
                                            "center_right"
                                        ) and intersection_info["center_right"].get(
                                            "detected"
                                        ):
                                            right_lane_available = True

                                        if (
                                            right_lane_available
                                            and intersection_info.get("lane_count", 0)
                                            >= 3
                                        ):
                                            converted_command = "turn_right"
                                        else:
                                            converted_command = (
                                                "straight"  # Fallback an toàn
                                            )

                                elif sign == "no_turn_right":
                                    # Cấm rẽ phải → KIỂM TRA ĐIỂM TRƯỚC để quyết định
                                    # Nếu điểm trước (y=200) vẫn có đường thẳng → đi thẳng
                                    # Nếu điểm trước không có đường thẳng → rẽ trái

                                    can_go_straight_ahead = (
                                        early_check_points is not None
                                    )

                                    if can_go_straight_ahead:
                                        # Vẫn còn đường phía trước → ưu tiên đi thẳng
                                        converted_command = "straight"
                                    else:
                                        # Không còn đường phía trước → phải rẽ
                                        left_lane_available = False

                                        if intersection_info.get(
                                            "left"
                                        ) and intersection_info["left"].get("detected"):
                                            left_lane_available = True
                                        elif intersection_info.get(
                                            "center_left"
                                        ) and intersection_info["center_left"].get(
                                            "detected"
                                        ):
                                            left_lane_available = True

                                        if (
                                            left_lane_available
                                            and intersection_info.get("lane_count", 0)
                                            >= 3
                                        ):
                                            converted_command = "turn_left"
                                        else:
                                            converted_command = (
                                                "straight"  # Fallback an toàn
                                            )

                                # Sử dụng lệnh đã chuyển đổi làm detected_sign
                                if converted_command:
                                    detected_sign = converted_command
                                    current_sign_found = True
                                break

                    # Cập nhật bộ lọc ổn định cho biển báo
                    traffic_sign_filter.update(current_sign_found, detected_sign)

                    # Kiểm tra biển báo có được phát hiện ổn định không
                    is_stable, stable_sign = (
                        traffic_sign_filter.is_sign_stably_detected()
                    )

                    # Cập nhật trạng thái biển báo dựa trên kết quả ổn định
                    if is_stable and not sign_detected:
                        # Biển báo được phát hiện ổn định, cập nhật trạng thái
                        sign_detected = True
                        traffic_sign_command = stable_sign
                        passed_sign = False
                        ready_to_turn = False
                        frames_without_sign = 0

                    # Kiểm tra đã đi qua biển báo chưa (sử dụng bộ lọc ổn định)
                    if (
                        sign_detected
                        and traffic_sign_filter.has_passed_sign()
                        and not passed_sign
                    ):
                        passed_sign = True
                        frames_without_sign = 0

                    # CHỈ sử dụng detect_lane_departure khi có biển báo lệnh hoặc cấm
                    # NHƯNG BỎ QUA khi có lệnh đi thẳng để tránh bị nhiễu
                    # SỬ DỤNG QUYẾT ĐỊNH ỔN ĐỊNH từ bộ lọc
                    suggested_turn = None
                    if (
                        has_command_sign or has_prohibition_sign or sign_detected
                    ) and traffic_sign_command != "straight":
                        # Sử dụng quyết định ổn định thay vì quyết định tức thời
                        if stable_left and avg_confidence > 50.6:
                            suggested_turn = "turn_left"
                        elif stable_right and avg_confidence > 0.65:
                            suggested_turn = "turn_right"

                    # Kiểm tra đã đi qua biển báo chưa
                    if sign_detected and not current_sign_found:
                        frames_without_sign += 1
                        # SỬ DỤNG FPS ADAPTER để tự động điều chỉnh
                        required_frames = fps_adapter.get_adapted_frames(
                            4
                        )  # Base: 4 frame cho 30FPS
                        if frames_without_sign >= required_frames and not passed_sign:
                            passed_sign = True
                            frames_without_sign = 0
                    else:
                        frames_without_sign = 0

                    # Kiểm tra sẵn sàng rẽ: YÊU CẦU NGHIÊM NGẶT HƠN
                    # CHỈ khi có biển báo lệnh VÀ ĐÃ QUA BIỂN BÁO VÀ đến giao lộ VÀ có đủ làn đường
                    # BỎ logic strong_turn_signal để tránh rẽ sớm

                    # Thêm điều kiện khoảng cách an toàn sau biển báo (SỬ DỤNG FPS ADAPTER)
                    safe_distance_after_sign = True
                    if passed_sign:
                        # SỬ DỤNG FPS ADAPTER: Base 5 frame cho 30FPS
                        if not hasattr(
                            detect_intersection_points, "frames_since_passed"
                        ):
                            detect_intersection_points.frames_since_passed = 0
                        detect_intersection_points.frames_since_passed += 1

                        required_safe_frames = fps_adapter.get_adapted_frames(5)
                        if (
                            detect_intersection_points.frames_since_passed
                            < required_safe_frames
                        ):
                            safe_distance_after_sign = False
                    else:
                        if hasattr(detect_intersection_points, "frames_since_passed"):
                            detect_intersection_points.frames_since_passed = 0

                    # Điều kiện backup: Cho phép rẽ nhanh hơn với tín hiệu mạnh và ổn định
                    strong_and_stable_signal = (
                        passed_sign  # Vẫn yêu cầu đã qua biển báo
                        and intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3
                        and stable_left
                        and avg_confidence > 0.8  # Tín hiệu rẽ trái mạnh
                        and traffic_sign_command == "turn_left"
                    ) or (
                        passed_sign  # Vẫn yêu cầu đã qua biển báo
                        and intersection_info["is_intersection"]
                        and intersection_info["lane_count"] >= 3
                        and stable_right
                        and avg_confidence > 0.8  # Tín hiệu rẽ phải mạnh
                        and traffic_sign_command == "turn_right"
                    )

                    if (
                        passed_sign  # BẮT BUỘC phải đã qua biển báo
                        and safe_distance_after_sign  # VÀ đã có khoảng cách an toàn (GIẢM)
                        and intersection_info["is_intersection"]  # VÀ đang ở giao lộ
                        and intersection_truly_stable  # VÀ giao lộ đã ổn định
                        and intersection_info["lane_count"]
                        >= 3  # GIẢM yêu cầu từ 4 xuống 3 làn đường
                        and not ready_to_turn
                        and traffic_sign_command
                        in ["turn_left", "turn_right", "straight"]
                    ) or (
                        strong_and_stable_signal  # HOẶC có tín hiệu mạnh và ổn định
                        and not ready_to_turn
                    ):
                        ready_to_turn = True
                        intersection_mode = True
                        # Khởi tạo bộ đếm frame intersection
                        if not hasattr(
                            detect_intersection_points, "intersection_frames"
                        ):
                            detect_intersection_points.intersection_frames = 0

                    # Đếm frame trong intersection mode
                    if intersection_mode and ready_to_turn:
                        if not hasattr(
                            detect_intersection_points, "intersection_frames"
                        ):
                            detect_intersection_points.intersection_frames = 0
                        detect_intersection_points.intersection_frames += 1
                    else:
                        if hasattr(detect_intersection_points, "intersection_frames"):
                            detect_intersection_points.intersection_frames = 0

                    # Thoát khỏi chế độ giao lộ khi đã rẽ xong (không còn ở giao lộ)
                    if intersection_mode and ready_to_turn:
                        if not intersection_info["is_intersection"]:
                            # Reset tất cả biến cờ khi thoát giao lộ
                            intersection_mode = False
                            traffic_sign_command = None
                            sign_detected = False
                            passed_sign = False
                            ready_to_turn = False
                            frames_without_sign = 0

                    # LOGIC RESET BỔ SUNG: Reset khi đã hoàn thành lệnh hoặc trong các trường hợp khác
                    reset_needed = False

                    # THÊM: Kiểm tra reset nhanh khi đã có lane rõ ràng sau khi rẽ
                    if intersection_mode and ready_to_turn and lane_points:
                        left_x, right_x, mid_x = lane_points
                        lane_width = right_x - left_x
                        lane_center_error = abs(mid_x - center_x)

                        # Nếu có lane rõ ràng và không còn ở giao lộ phức tạp
                        if (
                            lane_width > 60
                            and lane_center_error < 50
                            and intersection_info["lane_count"] <= 2
                        ):
                            reset_needed = True

                    # Trường hợp 1: RESET NGAY LẬP TỨC khi hoàn thành lệnh rẽ
                    if intersection_mode and ready_to_turn and traffic_sign_command:
                        # Kiểm tra đã hoàn thành lệnh rẽ dựa trên lane stability
                        if lane_points:
                            left_x, right_x, mid_x = lane_points
                            lane_width = right_x - left_x
                            lane_center_error = abs(mid_x - center_x)

                            # Nếu làn đường đã ổn định (error nhỏ và có lane rõ ràng) - NỚI LỎNG ĐỂ RESET DỄ HƠN CHO 60FPS
                            lane_stable = (
                                lane_center_error < 40  # NỚI LỎNG từ 35 lên 40
                                and lane_width > 70  # NỚI LỎNG từ 80 xuống 70
                            )

                            # Kiểm tra đã thực hiện đủ lệnh rẽ - GIẢM NGƯỎNG ĐỂ RESET SỚM HƠN CHO 60FPS
                            command_completed = False

                            if traffic_sign_command == "turn_left":
                                # Đã rẽ trái đủ nếu mid_x đã dịch sang trái - GIẢM NGƯỎNG CHO 60FPS
                                if mid_x < center_x - 30:  # GIẢM từ 40 xuống 30 pixel
                                    command_completed = True
                            elif traffic_sign_command == "turn_right":
                                # Đã rẽ phải đủ nếu mid_x đã dịch sang phải - GIẢM NGƯỎNG CHO 60FPS
                                if mid_x > center_x + 30:  # GIẢM từ 40 xuống 30 pixel
                                    command_completed = True
                            elif traffic_sign_command == "straight":
                                # Lệnh đi thẳng luôn hoàn thành nếu lane stable
                                command_completed = True

                            # Reset ngay nếu đã hoàn thành lệnh và lane ổn định
                            if command_completed and lane_stable:
                                reset_needed = True

                    # Trường hợp 1.5: Timeout trong intersection mode (tránh kẹt) - SỬ DỤNG FPS ADAPTER
                    if intersection_mode and ready_to_turn:
                        intersection_frames = getattr(
                            detect_intersection_points, "intersection_frames", 0
                        )
                        # SỬ DỤNG FPS ADAPTER: Base 25 frame cho 30FPS
                        timeout_frames = fps_adapter.get_adapted_frames(25)
                        if intersection_frames > timeout_frames:
                            reset_needed = True

                    # Trường hợp 2: Đã qua giao lộ (thực hiện xong lệnh)
                    if (
                        sign_detected or passed_sign or ready_to_turn
                    ) and not intersection_info["is_intersection"]:
                        # Nếu đã có trạng thái biển báo nhưng không còn ở giao lộ nữa
                        reset_needed = True

                    # Trường hợp 3: Thấy biển báo mới khác với biển báo cũ
                    if current_sign_found and traffic_sign_command:
                        for sign in current_traffic_signs:
                            if (
                                sign in ["turn_left", "turn_right", "straight"]
                                and sign != traffic_sign_command
                            ):
                                # Biển báo mới khác biển báo cũ
                                reset_needed = True
                                break

                    # Trường hợp 4: Đã qua biển báo quá lâu mà không đến giao lộ
                    if passed_sign and not intersection_info["is_intersection"]:
                        # Có thể thêm bộ đếm frame để reset sau một thời gian
                        if not hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0
                        detect_intersection_points.frames_after_sign += 1

                        # SỬ DỤNG FPS ADAPTER: Reset nếu đã đi quá base 40 frame mà không thấy giao lộ
                        timeout_after_sign = fps_adapter.get_adapted_frames(40)
                        if (
                            detect_intersection_points.frames_after_sign
                            > timeout_after_sign
                        ):
                            reset_needed = True
                            detect_intersection_points.frames_after_sign = 0
                    else:
                        if hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0

                    # Thực hiện reset nếu cần
                    if reset_needed:
                        intersection_mode = False
                        traffic_sign_command = None
                        sign_detected = False
                        passed_sign = False
                        ready_to_turn = False
                        frames_without_sign = 0
                        stability_filter.reset()  # Reset bộ lọc ổn định
                        traffic_sign_filter.reset()  # Reset bộ lọc biển báo
                        if hasattr(detect_intersection_points, "frames_after_sign"):
                            detect_intersection_points.frames_after_sign = 0
                        if hasattr(detect_intersection_points, "intersection_frames"):
                            detect_intersection_points.intersection_frames = 0
                        if hasattr(detect_intersection_points, "frames_since_passed"):
                            detect_intersection_points.frames_since_passed = 0
                        if hasattr(
                            detect_intersection_points, "intersection_stable_count"
                        ):
                            detect_intersection_points.intersection_stable_count = 0

                    img_resized = np.array(pil_image.resize((640, 360)))
                    mask_color = decode_segmap(pred_mask)
                    display_img = cv2.addWeighted(img_resized, 0.5, mask_color, 0.5, 0)

                    scan_line_y = 240
                    center_x = display_img.shape[1] // 2

                    # LOGIC ĐIỀU KHIỂN
                    if (
                        intersection_mode
                        and ready_to_turn
                        and traffic_sign_command
                        in ["turn_left", "turn_right", "straight"]
                    ):
                        # Chế độ giao lộ: CHỈ điều khiển theo biển báo lệnh (không phải unknown)
                        intersection_steer = calculate_intersection_steering(
                            intersection_info, traffic_sign_command
                        )
                        if intersection_steer is not None:
                            steer_angle = intersection_steer
                            car_speed = 80  # TĂNG từ 80->90 để có đủ lực leo dốc khi rẽ
                        else:
                            # Fallback về lane following nếu không tính được
                            if lane_points:
                                left_x, right_x, mid_x = lane_points
                                last_midpoint = mid_x
                                lane_error = mid_x - center_x
                            elif last_midpoint is not None:
                                lane_error = last_midpoint - center_x
                            steer_angle = lane_pid.compute(lane_error)
                            # THÊM: Phát hiện dốc và điều chỉnh tốc độ
                            slope_boost = detect_steep_slope(
                                lane_error, lane_points, prev_lane_points
                            )
                            base_speed = adaptive_speed_control(lane_error)
                            car_speed = int(base_speed * slope_boost)
                    else:
                        # Chế độ bình thường: CHỈ theo làn đường màu trắng
                        # (Không có biển báo lệnh, có biển unknown, hoặc chưa sẵn sàng rẽ)
                        if lane_points:
                            left_x, right_x, mid_x = lane_points
                            last_midpoint = mid_x
                            lane_error = mid_x - center_x
                        elif last_midpoint is not None:
                            lane_error = last_midpoint - center_x

                        steer_angle = lane_pid.compute(lane_error)
                        # THÊM: Phát hiện dốc và điều chỉnh tốc độ cho chế độ bình thường
                        slope_boost = detect_steep_slope(
                            lane_error, lane_points, prev_lane_points
                        )
                        base_speed = adaptive_speed_control(lane_error)
                        car_speed = int(base_speed * slope_boost)

                    # === DEBUG VISUALIZATION ===
                    # Tạo cửa sổ debug cho segmentation
                    debug_mask = mask_color.copy()

                    # Vẽ các điểm phát hiện giao lộ
                    scan_y_display = 360
                    for point_name, point_info in intersection_info_display.items():
                        if isinstance(point_info, dict) and point_info.get(
                            "detected", False
                        ):
                            center = point_info["center"]
                            color = {
                                "left": (255, 0, 0),  # Xanh dương cho trái
                                "center_left": (255, 255, 0),  # Cyan cho trái-giữa
                                "center": (0, 255, 0),  # Xanh lá cho giữa
                                "center_right": (0, 255, 255),  # Vàng cho phải-giữa
                                "right": (0, 0, 255),  # Đỏ cho phải
                            }.get(point_name, (255, 255, 255))

                            cv2.circle(
                                debug_mask, (center, scan_y_display), 8, color, -1
                            )
                            cv2.putText(
                                debug_mask,
                                point_name[0].upper(),
                                (center - 5, scan_y_display - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                    # Vẽ đường scan cho giao lộ
                    cv2.line(
                        debug_mask,
                        (0, scan_y_display),
                        (640, scan_y_display),
                        (128, 128, 128),
                        2,
                    )
                    cv2.putText(
                        debug_mask,
                        f"Intersection Check Y={scan_y_display}",
                        (10, scan_y_display - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 128),
                        1,
                    )

                    # Vẽ đường scan cho lane following
                    cv2.line(debug_mask, (0, 270), (640, 270), (255, 255, 0), 2)
                    cv2.putText(
                        debug_mask,
                        "Lane Following Y=270",
                        (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                    # Vẽ đường scan sớm (early check)
                    cv2.line(debug_mask, (0, 200), (640, 200), (0, 255, 0), 2)
                    cv2.putText(
                        debug_mask,
                        "Early Check Y=200",
                        (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    # Vẽ lane points hiện tại
                    if lane_points:
                        left_x, right_x, mid_x = lane_points
                        cv2.circle(debug_mask, (mid_x, 270), 6, (255, 255, 255), -1)
                        cv2.circle(debug_mask, (left_x, 270), 4, (255, 0, 0), -1)
                        cv2.circle(debug_mask, (right_x, 270), 4, (255, 0, 0), -1)
                        # Vẽ error line
                        cv2.line(
                            debug_mask,
                            (center_x, 270),
                            (mid_x, 270),
                            (255, 255, 255),
                            3,
                        )
                        cv2.putText(
                            debug_mask,
                            f"Error: {mid_x-center_x}",
                            (mid_x + 10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                    # Vẽ center line
                    cv2.line(
                        debug_mask, (center_x, 0), (center_x, 360), (128, 128, 128), 1
                    )

                    # === YOLO DEBUG ===
                    cam_debug = cam_image.copy()

                    # Vẽ bounding boxes cho traffic signs
                    for detection in detections:
                        bbox = detection["bbox"]
                        class_id = detection["class"]
                        confidence = detection["confidence"]
                        area = detection["area"]

                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 255, 0) if area >= 500 else (0, 0, 255)
                        cv2.rectangle(cam_debug, (x1, y1), (x2, y2), color, 2)

                        label = (
                            f"{YOLO_CLASSES[class_id]} {confidence:.2f} A:{int(area)}"
                        )
                        cv2.putText(
                            cam_debug,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )
                    # ✅ Vẽ khung đỏ lên camera khi thấy xe/người
                    for (x1, y1, x2, y2, label, conf) in vehicle_boxes:
                        cv2.rectangle(cam_debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                                cam_debug, f"{label.upper()} {conf:.2f}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )
                    # --- 🛑 STOP khi phát hiện xe khác (vehicle box) ---
                    VEHICLE_STOP_TIME =time_stop  # dừng 3 giây

                    # ✅ đảm bảo biến stop_until tồn tại
                    if "stop_until" not in locals():
                        stop_until = None

                    if len(vehicle_boxes) > 0:  # nếu có xe khác trong khung
                        if stop_until is None or time.time() > stop_until:
                            stop_until = time.time() + VEHICLE_STOP_TIME

                    stop_active = (stop_until is not None and time.time() < stop_until)

                    if stop_active:
                        steer_angle = 0
                        car_speed = 0
                        try:
                            socket.sendall(bytes(f"{steer_angle} {car_speed}", "utf-8"))
                        except Exception:
                            pass    
                    
        


                    # Thêm thông tin state lên video
                    info_lines = [
                        f"State: {'INTERSECTION' if intersection_mode else 'NORMAL'}",
                        f"Speed: {car_speed}",
                        f"Steer: {steer_angle}",
                        f"Command: {traffic_sign_command or 'None'}",
                        f"Sign Detected: {sign_detected}",
                        f"Passed Sign: {passed_sign}",
                        f"Ready to Turn: {ready_to_turn}",
                        f"Lane Count: {intersection_info.get('lane_count', 0)}",
                        f"Is Intersection: {intersection_info.get('is_intersection', False)}",
                        f"Stable Left: {stable_left}",
                        f"Stable Right: {stable_right}",
                        f"Confidence: {avg_confidence:.2f}",
                        f"Current Signs: {current_traffic_signs}",
                    ]

                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            cam_debug,
                            line,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                    # Hiển thị cửa sổ debug
                    cv2.imshow("Traffic Signs Detection", cam_debug)
                    cv2.imshow("Lane Segmentation Debug", debug_mask)

                    # Thoát nếu nhấn 'q' hoặc ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 27 là ESC
                        break

                    # Cập nhật prev_lane_points cho frame tiếp theo
                    prev_lane_points = lane_points

                except Exception as e:
                    print(f"Error in processing: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
            socket.close()


# Chạy chương trình
if __name__ == "__main__":
    print("Starting lane following + traffic sign detection...")
    print("Press ESC to exit")