#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FULL: V11 normal + V9 bridge (auto switch) + Lane PID + Sign/Intersection Turn-Commit
#      + Traffic Light (robust stable+latch) + IMU Bridge v2 policy.

import os
import cv2
import time
import numpy as np
from collections import deque
from rknn.api import RKNN

from utils import preprocess, postprocess, gen_color, vis_result
from control import control, close_serial

# ================== IMU + BRIDGE ==================
try:
    from bno055_uart import BNO055Compass, BridgeProcessor
except Exception:
    BNO055Compass = None
    BridgeProcessor = None

# ================== 2 MODEL PATHS ==================
RKNN_PATH_V11 = '/home/orangepi/CDS_UTE_2025/best_v11.rknn'
RKNN_PATH_V9  = '/home/orangepi/CDS_UTE_2025/best_9_class.rknn'

INPUT_W = 320
INPUT_H = 320

CONF_THRES = 0.10
IOU_THRES  = 0.45

# Road/sign threshold per model
CONF_THRES_ROAD_V11 = 0.80
CONF_THRES_ROAD_V9  = 0.15
CONF_SIGN_V11 = 0.90
CONF_SIGN_V9  = 0.90

# ✅ IMPORTANT: CLASSES must match EXACT order of each RKNN
# If V9 has different order/count, edit CLASSES_V9 accordingly.
CLASSES = [  ### đổi chỗ  right_left
    "brige",
    "go_straight",
    "no_left",
    "no_right",
    "no_straight",
    "road",
    "stop_sign",
    "traffic_light",
    "turn_left",
    "turn_right",
]

ROAD_CLASS_NAME   = 'road'
TLIGHT_CLASS_NAME = 'traffic_light'
BRIDGE_LABEL_NAME = 'brige'

# ================== CAMERA ==================
MIRROR_CAMERA = False  # True nếu camera bị mirror trái-phải

# ================== LANE & PID ==================
PATH_TOP_RATIO    = 0.10
PATH_BOTTOM_RATIO = 0.60
PATH_ROW_STEP     = 3

KP = 0.275
KI = 0.0005
KD = 0.030
MAX_ANGLE = 30

BASE_SPEED      = 25
MIN_SPEED       = 20
ERROR_NORM_PIX  = 80

MAX_LOST_FRAMES = 8

CURV_THRESH_1   = 20.0
CURV_THRESH_2   = 40.0
CURV_ERROR_BOOST_1 = 1.2
CURV_ERROR_BOOST_2 = 1.4

# ================== SIGN ==================
SIGN_HOLD_FRAMES = 15
TURN_BIAS_PIX    = 40
NO_TURN_TOL      = 20

MIN_SIGN_AREA    = 25 * 25
SIGN_ROI_Y_MIN   = 0.30
SIGN_ROI_Y_MAX   = 0.80
SIGN_ROI_X_MIN   = 0.35
SIGN_ROI_X_MAX   = 0.65

# ================== TRAFFIC LIGHT ==================
TLIGHT_DET_CONF_LOW  = 0.25
TLIGHT_DET_CONF_HIGH = 0.50
TLIGHT_MIN_AREA      = 19 * 19

TLIGHT_HISTORY_LEN   = 10
TLIGHT_MIN_STABLE    = 3
TLIGHT_ACCEPT_FRAC   = 0.55
TLIGHT_RED_LATCH_FRAMES = 60
TLIGHT_APPROACH_SPEED = 6
TLIGHT_GREEN_GO_DELAY_FRAMES = 20
TLIGHT_AFTER_GREEN_IGNORE_RED_FRAMES = 15

# ================== INTERSECTION + TURN COMMIT ==================
SIGN_MEMORY_TTL_FRAMES = 60

INTERSECT_BAND_Y0 = 0.28
INTERSECT_BAND_Y1 = 0.62
INTERSECT_ROW_STEP = 6

INTERSECT_WIDTH_RATIO_THRESH = 0.78
INTERSECT_SEGMENTS_THRESH    = 2
DIR_STRENGTH_MIN = 1200

TURN_COMMIT_FRAMES = 26
TURN_RAMP_FRAMES   = 8
TURN_TARGET_OFFSET_RATIO = 0.30
TURN_SPEED = 18

# ================== RECOVERY (reverse only) ==================
MODE_FORWARD    = 0
MODE_RECOV_BACK = 1
RECOV_LOST_FRAMES_TRIGGER = 2
RECOV_REVERSE_SPEED  = -12

# ================== TRAPEZOID ROI ==================
TRAP_TOP_Y_RATIO    = 0.25
TRAP_BOTTOM_Y_RATIO = 0.95
TRAP_TOP_WIDTH_RATIO    = 0.35
TRAP_BOTTOM_WIDTH_RATIO = 1.00

# ================== BRIDGE POLICY (from your V9 bridge code) ==================
BRIDGE_TOP_SLOW_COEFF   = 0.65
BRIDGE_DOWN_SLOW_COEFF  = 0.15

BRIDGE_DOWN_ANGLE_COEFF = 0.6
BRIDGE_DOWN_MAX_ANGLE   = 18

BRIDGE_DOWN_STOP_HOLD_FR      = 10
BRIDGE_DOWN_STOP1_AT_FR       = 8
BRIDGE_DOWN_END_PITCH_DEG     = 3.0
BRIDGE_DOWN_MIN_FR_BEFORE_END = 12
BRIDGE_DOWN_FALLBACK_FR       = 3

# ================== VIDEO ==================
SAVE_VIDEO = True
VIDEO_DIR  = '/home/orangepi/CDS_UTE_2025/log_videos'
VIDEO_FPS  = 20.0


# ================== LANE HELPERS ==================
def compute_path_center(mask, top_ratio=PATH_TOP_RATIO, bottom_ratio=PATH_BOTTOM_RATIO,
                        step=PATH_ROW_STEP, min_total_pixels=200):
    h, w = mask.shape
    y0 = int(h * top_ratio)
    y1 = int(h * bottom_ratio)
    if y1 <= y0:
        y1 = y0 + 1

    rows = []
    total_pixels = 0

    for y in range(y0, y1, step):
        row = mask[y, :]
        xs = np.where(row > 0)[0]
        if xs.size < 5:
            continue
        left = int(xs[0])
        right = int(xs[-1])
        cx_row = 0.5 * (left + right)
        rows.append((y, cx_row))
        total_pixels += xs.size

    if len(rows) == 0 or total_pixels < min_total_pixels:
        return None, None, None

    rows.sort(key=lambda r: r[0])
    far_cx  = rows[0][1]
    near_cx = rows[-1][1]
    path_cx = 0.3 * far_cx + 0.7 * near_cx
    return float(path_cx), float(near_cx), float(far_cx)


def keep_largest_lane(mask, min_area=300):
    if mask is None:
        return mask
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return m
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx_relative = int(np.argmax(areas))
    max_area = areas[max_idx_relative]
    if max_area < min_area:
        return np.zeros_like(mask, dtype=np.uint8)
    max_idx = 1 + max_idx_relative
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == max_idx] = 1
    return out


def smooth_lane_edges(mask, erode_iter=1, dilate_iter=1, kernel_size=3):
    if mask is None or mask.max() == 0:
        return mask
    m = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    if erode_iter > 0:
        m = cv2.erode(m, kernel, iterations=erode_iter)
    if dilate_iter > 0:
        m = cv2.dilate(m, kernel, iterations=dilate_iter)
    return (m > 0).astype(np.uint8)


def apply_trapezoid_roi(mask,
                        top_y_ratio=TRAP_TOP_Y_RATIO,
                        bottom_y_ratio=TRAP_BOTTOM_Y_RATIO,
                        top_width_ratio=TRAP_TOP_WIDTH_RATIO,
                        bottom_width_ratio=TRAP_BOTTOM_WIDTH_RATIO):
    if mask is None or mask.max() == 0:
        return mask
    h, w = mask.shape
    y_top = int(h * top_y_ratio)
    y_bottom = int(h * bottom_y_ratio)
    if y_bottom <= y_top:
        y_bottom = y_top + 1

    cx = w // 2
    half_top = int((w * top_width_ratio) / 2.0)
    half_bottom = int((w * bottom_width_ratio) / 2.0)

    x_left_top     = int(np.clip(cx - half_top, 0, w - 1))
    x_right_top    = int(np.clip(cx + half_top, 0, w - 1))
    x_left_bottom  = int(np.clip(cx - half_bottom, 0, w - 1))
    x_right_bottom = int(np.clip(cx + half_bottom, 0, w - 1))

    pts = np.array([
        [x_left_bottom,  y_bottom],
        [x_right_bottom, y_bottom],
        [x_right_top,    y_top],
        [x_left_top,     y_top],
    ], dtype=np.int32)

    roi = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(roi, [pts], 1)
    return mask * roi


def get_left_right_lane_strength(mask, cx):
    h, w = mask.shape
    cx = int(np.clip(cx, 0, w - 1))
    left_area  = int(mask[:, :cx].sum())
    right_area = int(mask[:, cx:].sum())
    return left_area, right_area


def get_lane_from_road_boxes(road_boxes):
    if not road_boxes:
        return None
    best_box = max(road_boxes, key=lambda b: float(b[4]))
    x1, y1, x2, y2, conf, cls_id = best_box
    return float((x1 + x2) / 2.0)


# ================== INTERSECTION HELPERS ==================
def _count_row_segments(row_bin):
    r = (row_bin > 0).astype(np.uint8)
    if r.max() == 0:
        return 0
    diff = np.diff(r, prepend=0)
    return int(np.sum(diff == 1))


def detect_intersection(lane_mask_full):
    if lane_mask_full is None or lane_mask_full.max() == 0:
        return False
    h, w = lane_mask_full.shape
    y0 = int(h * INTERSECT_BAND_Y0)
    y1 = int(h * INTERSECT_BAND_Y1)
    if y1 <= y0:
        return False

    max_width_ratio = 0.0
    max_segments = 0
    for y in range(y0, y1, INTERSECT_ROW_STEP):
        row = lane_mask_full[y, :]
        xs = np.where(row > 0)[0]
        if xs.size < 10:
            continue
        width = float(xs[-1] - xs[0])
        max_width_ratio = max(max_width_ratio, width / float(w))
        max_segments = max(max_segments, _count_row_segments(row))

    return (max_width_ratio >= INTERSECT_WIDTH_RATIO_THRESH) or (max_segments >= INTERSECT_SEGMENTS_THRESH)


def _row_segments_centers(row_bin):
    xs = np.where(row_bin > 0)[0]
    if xs.size < 10:
        return []
    gaps = np.where(np.diff(xs) > 3)[0]
    segs = []
    start = 0
    for g in gaps:
        seg = xs[start:g+1]
        segs.append((int(seg[0]), int(seg[-1])))
        start = g + 1
    seg = xs[start:]
    segs.append((int(seg[0]), int(seg[-1])))

    centers = [0.5 * (a + b) for a, b in segs if (b - a) >= 8]
    return centers


def branch_targets_from_mask(lane_mask):
    if lane_mask is None or lane_mask.max() == 0:
        return None
    h, w = lane_mask.shape
    cx0 = w * 0.5
    y0 = int(h * 0.55)
    y1 = int(h * 0.85)
    if y1 <= y0:
        y0 = int(h * 0.45)
        y1 = int(h * 0.80)

    left_list, right_list, center_list = [], [], []
    for y in range(y0, y1, 6):
        centers = _row_segments_centers(lane_mask[y, :])
        if not centers:
            continue
        centers = sorted(centers)
        left_list.append(centers[0])
        right_list.append(centers[-1])
        c = min(centers, key=lambda c: abs(c - cx0))
        center_list.append(c)

    if len(center_list) < 3:
        return None

    def med(a):
        a = sorted(a)
        return float(a[len(a)//2])

    return {
        'left': med(left_list) if left_list else None,
        'right': med(right_list) if right_list else None,
        'straight': med(center_list),
    }


def direction_strengths(lane_mask):
    if lane_mask is None:
        return 0, 0, 0
    h, w = lane_mask.shape
    y0 = int(h * INTERSECT_BAND_Y0)
    y1 = int(h * INTERSECT_BAND_Y1)
    band = lane_mask[y0:y1, :] if y1 > y0 else lane_mask
    third = w // 3
    left  = int(band[:, :third].sum())
    cent  = int(band[:, third:2*third].sum())
    right = int(band[:, 2*third:].sum())
    return left, cent, right


def decide_maneuver_from_sign(sign_label, lane_mask_full):
    if sign_label is None:
        return None

    left_s, cent_s, right_s = direction_strengths(lane_mask_full)

    def has_left():  return left_s  >= DIR_STRENGTH_MIN
    def has_cent():  return cent_s  >= DIR_STRENGTH_MIN
    def has_right(): return right_s >= DIR_STRENGTH_MIN

    if sign_label == 'turn_left':
        return 'left'
    if sign_label == 'turn_right':
        return 'right'
    if sign_label == 'go_straight':
        return 'straight'

    if sign_label == 'no_left':
        if has_cent():  return 'straight'
        if has_right(): return 'right'
        return 'straight'

    if sign_label == 'no_right':
        if has_cent():  return 'straight'
        if has_left():  return 'left'
        return 'straight'

    if sign_label == 'no_straight':
        if has_left() and left_s >= right_s * 1.05:
            return 'left'
        if has_right() and right_s >= left_s * 1.05:
            return 'right'
        return 'left' if left_s >= right_s else 'right'

    if sign_label == 'stop_sign':
        return 'straight'

    return None


# ================== PID ==================
class PIDController:
    def __init__(self, kp, ki, kd, max_output=25):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt < 1e-3:
            dt = 1e-3

        e = float(error)
        p = self.kp * e
        self.integral += e * dt
        self.integral = max(min(self.integral, 8.0), -8.0)
        i = self.ki * self.integral

        de = (e - self.last_error) / dt
        self.last_error = e
        d = self.kd * de

        out = p + i + d
        out = max(min(out, self.max_output), -self.max_output)
        return float(out)


def adaptive_speed_control(error_px, base_speed=BASE_SPEED, min_speed=MIN_SPEED):
    e = abs(float(error_px))
    x = min(e / ERROR_NORM_PIX, 1.5)
    factor = 1.0 - x ** 1.6
    speed = base_speed * max(factor, 0.0)
    if speed < min_speed:
        speed = min_speed
    return int(speed)


# ================== SIGN STATE + MEMORY ==================
class SignState:
    def __init__(self, hold_frames=SIGN_HOLD_FRAMES,
                 history_len=15, min_stable=4,
                 stable_frac_accept=0.7, stable_frac_switch=0.85):
        self.label = None
        self.frames_left = 0
        self.hold_frames = hold_frames
        self.history = deque(maxlen=history_len)
        self.min_stable = min_stable
        self.stable_frac_accept = stable_frac_accept
        self.stable_frac_switch = stable_frac_switch

    def step(self):
        if self.frames_left > 0:
            self.frames_left -= 1
            if self.frames_left == 0:
                self.label = None

    def update_from_dets(self, det_labels, det_confs):
        VALID_SIGNS = (
            'go_straight', 'turn_left', 'turn_right',
            'no_left', 'no_right', 'no_straight',
            'stop_sign',
        )

        best_label = None
        best_conf = 0.0
        for lab, cf in zip(det_labels, det_confs):
            if lab not in VALID_SIGNS:
                continue
            if cf > best_conf:
                best_conf = cf
                best_label = lab

        self.history.append(best_label)

        counts = {}
        total_non_none = 0
        for lab in self.history:
            if lab is None:
                continue
            counts[lab] = counts.get(lab, 0) + 1
            total_non_none += 1

        if total_non_none < self.min_stable or not counts:
            return

        stable_label = max(counts, key=counts.get)
        stable_frac = counts[stable_label] / float(total_non_none)

        if stable_frac < self.stable_frac_accept:
            return

        if self.label is None or stable_label == self.label:
            self.label = stable_label
            self.frames_left = self.hold_frames
        else:
            if stable_frac >= self.stable_frac_switch:
                self.label = stable_label
                self.frames_left = self.hold_frames

    def current(self):
        return self.label


class SignMemory:
    def __init__(self, ttl_frames=SIGN_MEMORY_TTL_FRAMES):
        self.ttl = int(ttl_frames)
        self.label = None
        self.left = 0

    def update(self, label):
        if label is None:
            return
        if label in ('turn_left','turn_right','go_straight','no_left','no_right','no_straight','stop_sign'):
            self.label = label
            self.left = self.ttl

    def step(self):
        if self.left > 0:
            self.left -= 1
            if self.left <= 0:
                self.label = None
                self.left = 0

    def get(self):
        return self.label

    def consume(self):
        self.label = None
        self.left = 0


class TurnDecisionState:
    def __init__(self):
        self.maneuver = None
        self.left = 0
        self.total = 0
        self.offset_px = 0

    def is_active(self):
        return self.left > 0 and self.maneuver is not None

    def start(self, maneuver, img_width, target_cx=None):
        if maneuver is None:
            return
        self.maneuver = maneuver
        self.left = int(TURN_COMMIT_FRAMES)
        self.total = int(TURN_COMMIT_FRAMES)

        cx0 = img_width * 0.5
        if target_cx is None:
            off = img_width * TURN_TARGET_OFFSET_RATIO
            if maneuver == 'left':
                target_cx = cx0 - off
            elif maneuver == 'right':
                target_cx = cx0 + off
            else:
                target_cx = cx0

        self.offset_px = int(target_cx - cx0)

    def step(self):
        if self.left > 0:
            self.left -= 1
            if self.left <= 0:
                self.maneuver = None
                self.left = 0
                self.total = 0
                self.offset_px = 0

    def apply(self, lane_error_px):
        if not self.is_active():
            return float(lane_error_px)
        elapsed = (self.total - self.left)
        alpha = min(1.0, elapsed / float(max(TURN_RAMP_FRAMES, 1)))
        target_error = float(self.offset_px)
        lane_error = float(lane_error_px)
        return (1.0 - alpha) * lane_error + alpha * target_error

    def current(self):
        return self.maneuver


def apply_sign_logic(error_px, lane_mask, sign_label, cx):
    if sign_label is None:
        return float(error_px)
    e = float(error_px)

    if sign_label == 'go_straight':
        return e * 0.5
    if sign_label == 'turn_left':
        return e - TURN_BIAS_PIX
    if sign_label == 'turn_right':
        return e + TURN_BIAS_PIX

    if sign_label == 'no_left':
        if e < -NO_TURN_TOL:
            e = max(e, 0.0)
        return e

    if sign_label == 'no_right':
        if e > NO_TURN_TOL:
            e = min(e, 0.0)
        return e

    if sign_label == 'no_straight':
        left_area, right_area = get_left_right_lane_strength(lane_mask, cx)
        if left_area > right_area * 1.1:
            return e - TURN_BIAS_PIX
        elif right_area > left_area * 1.1:
            return e + TURN_BIAS_PIX
        else:
            return e - TURN_BIAS_PIX * 0.7

    return e


# ================== TRAFFIC LIGHT COLOR ==================
def _hsv_vote(hsv):
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_green = np.array([35, 70, 70])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_cnt = int(cv2.countNonZero(mask_red))
    green_cnt = int(cv2.countNonZero(mask_green))
    total = hsv.shape[0] * hsv.shape[1]
    if total <= 0:
        return None, 0.0, 0.0

    red_ratio = red_cnt / total
    green_ratio = green_cnt / total
    return (red_cnt, green_cnt), red_ratio, green_ratio


def classify_traffic_light_color_robust(img_unknown, box):
    if img_unknown is None or box is None:
        return None

    x1, y1, x2, y2 = box[:4]
    h, w = img_unknown.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = img_unknown[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    hsv_bgr = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, rr_bgr, gr_bgr = _hsv_vote(hsv_bgr)

    hsv_rgb = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    _, rr_rgb, gr_rgb = _hsv_vote(hsv_rgb)

    def score(rr, gr):
        return max(rr, gr)

    sb = score(rr_bgr, gr_bgr)
    sr = score(rr_rgb, gr_rgb)
    rr, gr = (rr_bgr, gr_bgr) if sb >= sr else (rr_rgb, gr_rgb)

    MIN_COLOR_RATIO = 0.02
    if rr < MIN_COLOR_RATIO and gr < MIN_COLOR_RATIO:
        return None

    if rr > gr * 1.2:
        return 'red'
    if gr > rr * 1.2:
        return 'green'
    return None


class TrafficLightState:
    def __init__(self,
                 history_len=TLIGHT_HISTORY_LEN,
                 min_stable=TLIGHT_MIN_STABLE,
                 accept_frac=TLIGHT_ACCEPT_FRAC,
                 red_latch_frames=TLIGHT_RED_LATCH_FRAMES):
        self.history = deque(maxlen=history_len)
        self.min_stable = min_stable
        self.accept_frac = accept_frac
        self.red_latch_frames = int(red_latch_frames)

        self.label = None
        self.red_latch_left = 0

    def update(self, color_label):
        self.history.append(color_label)

        counts = {'red': 0, 'green': 0}
        total = 0
        for c in self.history:
            if c in counts:
                counts[c] += 1
                total += 1

        stable_label = None
        stable_frac = 0.0
        if total >= self.min_stable:
            stable_label = 'red' if counts['red'] >= counts['green'] else 'green'
            stable_frac = counts[stable_label] / float(total)

        if stable_label == 'red' and stable_frac >= self.accept_frac:
            self.label = 'red'
            self.red_latch_left = self.red_latch_frames
            return

        if stable_label == 'green' and stable_frac >= self.accept_frac:
            self.label = 'green'
            self.red_latch_left = 0
            return

        if self.red_latch_left > 0:
            self.label = 'red'
            self.red_latch_left -= 1
        else:
            self.label = None

    def current(self):
        return self.label

    def reset(self):
        self.history.clear()
        self.label = None
        self.red_latch_left = 0


# ================== RKNN LOAD ==================
def load_rknn(path, target='rk3588'):
    r = RKNN()
    ret = r.load_rknn(path)
    if ret != 0:
        return None, ret
    ret = r.init_runtime(target=target)
    if ret != 0:
        try:
            r.release()
        except Exception:
            pass
        return None, ret
    return r, 0


# ================== MAIN ==================
def main():
    print("FULL: V11 normal + V9 bridge(auto switch) + lane PID + signs + intersection + traffic light + IMU bridge")

    # load two models
    print("[INIT] Loading V11...")
    rknn_v11, ret = load_rknn(RKNN_PATH_V11)
    if ret != 0 or rknn_v11 is None:
        print(f"❌ Load/init V11 failed, code={ret}")
        return
    print("✅ V11 ready")

    print("[INIT] Loading V9...")
    rknn_v9, ret = load_rknn(RKNN_PATH_V9)
    if ret != 0 or rknn_v9 is None:
        print(f"❌ Load/init V9 failed, code={ret}")
        try:
            rknn_v11.release()
        except Exception:
            pass
        return
    print("✅ V9 ready")

    # camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        rknn_v11.release()
        rknn_v9.release()
        return

    # video writer
    writer = None
    video_path = None
    if SAVE_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(VIDEO_DIR, f'drive_{ts}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (INPUT_W * 3, INPUT_H))
        if writer.isOpened():
            print("Recording:", video_path)
        else:
            print("⚠️ VideoWriter failed, no recording.")
            writer = None

    # controllers
    pid = PIDController(KP, KI, KD, MAX_ANGLE)
    sign_state = SignState()
    sign_mem = SignMemory()
    turn_state = TurnDecisionState()

    tlight_state = TrafficLightState()
    prev_tl = None
    green_delay_left = 0
    green_protect_left = 0

    # IMU
    imu = BNO055Compass() if BNO055Compass is not None else None
    bridge_proc = BridgeProcessor() if BridgeProcessor is not None else None

    bridge_armed = False
    bridge_seen_count = 0
    last_bridge_frame = -9999
    bridge2_seen = False
    top_slow_active = False

    bridge_state_name = "NORMAL"
    prev_bridge_state_name = "NORMAL"
    pitch_raw = 0.0
    pitch_filtered = 0.0
    bridge_speed_coeff = 1.0

    down_frames = 0
    down_stop_frames_left = 0
    down_stop1_done = False
    down_stop2_done = False

    # perf/state
    fps_hist = deque(maxlen=50)
    frame_count = 0

    lost_frames = 0
    mode = MODE_FORWARD
    angle_history = deque(maxlen=20)
    last_good_angle_cmd = 0.0
    base_turn_angle = 0.0

    # det-hold per model (avoid mixing)
    prev_det = {
        "V11": {"boxes": None, "masks": None, "shape": None, "hold": 0},
        "V9":  {"boxes": None, "masks": None, "shape": None, "hold": 0},
    }
    MAX_DET_HOLD = 5

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("Camera frame lost.")
                break

            # ================= IMU UPDATE =================
            prev_bridge_state_name = bridge_state_name
            if imu is not None and bridge_proc is not None:
                euler = imu.get_euler_angles()
                if euler:
                    _, _, pitch_raw = euler
                    imu_ctrl = bridge_proc.update(pitch_raw)
                    bridge_state_name  = imu_ctrl.get("state", "NORMAL")
                    pitch_filtered     = imu_ctrl.get("pitch_filtered", 0.0)
                    bridge_speed_coeff = imu_ctrl.get("speed_coeff", 1.0)
                else:
                    bridge_state_name = "NORMAL"
                    pitch_filtered = 0.0
                    bridge_speed_coeff = 1.0
            else:
                bridge_state_name = "NORMAL"
                pitch_filtered = 0.0
                bridge_speed_coeff = 1.0

            # DOWN counters
            if bridge_state_name != "DOWN":
                down_frames = 0
                down_stop_frames_left = 0
                down_stop1_done = False
                down_stop2_done = False
            else:
                down_frames += 1

            # End bridge (DOWN -> NORMAL)
            if prev_bridge_state_name == "DOWN" and bridge_state_name == "NORMAL":
                bridge_armed = False
                bridge_seen_count = 0
                bridge2_seen = False
                top_slow_active = False
                down_frames = 0
                down_stop_frames_left = 0
                down_stop1_done = False
                down_stop2_done = False
                print("[BRIDGE] END -> reset")

            # ================= PREPROCESS =================
            img0 = frame.copy()
            img4, img3 = preprocess(img0, INPUT_H, INPUT_W)
            img_in = img3.astype(np.uint8)

            # ================= MODEL SWITCH =================
            bridge_active = (bridge_state_name in ("UP", "TOP", "DOWN")) or bridge_armed
            if bridge_active:
                rknn_use = rknn_v9
                
                CONF_THRES_ROAD = CONF_THRES_ROAD_V9
                CONF_SIGN = CONF_SIGN_V9
                active_model_name = "V9"
            else:
                rknn_use = rknn_v11
                
                CONF_THRES_ROAD = CONF_THRES_ROAD_V11
                CONF_SIGN = CONF_SIGN_V11
                active_model_name = "V11"

            # ================= INFERENCE =================
            try:
                outputs = rknn_use.inference(inputs=[img_in], data_format='nhwc')
            except KeyboardInterrupt:
                print("[STOP] KeyboardInterrupt -> (0,0)")
                control(0, 0)
                break
            except Exception as e:
                print(f"[ERROR] inference: {e} -> (0,0)")
                control(0, 0)
                continue

            if outputs is None:
                print("[ERROR] inference None -> (0,0)")
                control(0, 0)
                continue

            results_list = postprocess(
                outputs, img4, img3,
                CONF_THRES, IOU_THRES,
                classes=len(CLASSES),
            )
            boxes, masks, shape_hw = results_list[0]

            # ================= DET-HOLD per model =================
            key = active_model_name
            has_det_now = isinstance(boxes, np.ndarray) and boxes.size > 0 and \
                          isinstance(masks, np.ndarray) and masks.size > 0

            if has_det_now:
                prev_det[key]["boxes"] = boxes
                prev_det[key]["masks"] = masks
                prev_det[key]["shape"] = shape_hw
                prev_det[key]["hold"] = 0
            else:
                if prev_det[key]["boxes"] is not None and prev_det[key]["hold"] < MAX_DET_HOLD:
                    boxes = prev_det[key]["boxes"]
                    masks = prev_det[key]["masks"]
                    shape_hw = prev_det[key]["shape"]
                    prev_det[key]["hold"] += 1
                else:
                    prev_det[key]["boxes"] = None
                    prev_det[key]["masks"] = None
                    prev_det[key]["shape"] = None
                    prev_det[key]["hold"] = 0

            if isinstance(shape_hw, (tuple, list)) and len(shape_hw) == 2:
                H, W = shape_hw
            else:
                H, W = INPUT_H, INPUT_W

            # ================= PARSE DETECTIONS =================
            lane_mask = np.zeros((H, W), dtype=np.uint8)
            lane_mask_full = None

            sign_labels = []
            sign_confs = []
            road_boxes = []
            tlight_candidates = []

            bridge_detected = False

            if isinstance(boxes, np.ndarray) and boxes.size > 0 and \
               isinstance(masks, np.ndarray) and masks.size > 0:

                for box, mask in zip(boxes, masks):
                    cls_id = int(box[5])
                    conf = float(box[4])
                    if cls_id < 0 or cls_id >= len(CLASSES):
                        continue
                    name = CLASSES[cls_id]

                    if MIRROR_CAMERA:
                        if name == 'turn_left':  name = 'turn_right'
                        elif name == 'turn_right': name = 'turn_left'
                        elif name == 'no_left': name = 'no_right'
                        elif name == 'no_right': name = 'no_left'

                    x1, y1, x2, y2 = box[:4]
                    w_box = max(x2 - x1, 0.0)
                    h_box = max(y2 - y1, 0.0)
                    area = w_box * h_box
                    cx_box = (x1 + x2) / 2.0
                    cy_box = (y1 + y2) / 2.0

                    # ROAD
                    if name == ROAD_CLASS_NAME:
                        if conf < CONF_THRES_ROAD:
                            continue
                        lane_mask |= (mask > 0).astype(np.uint8)
                        road_boxes.append(box)
                        continue

                    # BRIDGE SIGN
                    if name == BRIDGE_LABEL_NAME:
                        if conf >= CONF_SIGN and area >= MIN_SIGN_AREA:
                            bridge_detected = True
                        continue

                    # TLIGHT (only if class exists)
                    if name == TLIGHT_CLASS_NAME:
                        if area >= TLIGHT_MIN_AREA and conf >= TLIGHT_DET_CONF_LOW:
                            tlight_candidates.append((box, conf, area))
                        continue

                    # OTHER SIGNS
                    if conf < CONF_SIGN or area < MIN_SIGN_AREA:
                        continue
                    cx_norm = cx_box / float(W)
                    cy_norm = cy_box / float(H)
                    if cx_norm < SIGN_ROI_X_MIN or cx_norm > SIGN_ROI_X_MAX:
                        continue
                    if cy_norm < SIGN_ROI_Y_MIN or cy_norm > SIGN_ROI_Y_MAX:
                        continue

                    sign_labels.append(name)
                    sign_confs.append(conf)

            # ================= BRIDGE SIGN COUNT =================
            if bridge_detected:
                if frame_count - last_bridge_frame > 30:
                    last_bridge_frame = frame_count
                    if not bridge_armed:
                        bridge_armed = True
                        bridge_seen_count = 1
                        print("[BRIDGE] brige1 -> ARM")
                    else:
                        bridge_seen_count += 1
                        print(f"[BRIDGE] brige again count={bridge_seen_count}")

                    if bridge_seen_count >= 2:
                        bridge2_seen = True
                        print("[BRIDGE] brige2 (MID) -> allow TOP slow + prefer DOWN policy")

            # TOP slow activation
            if bridge_armed and bridge2_seen and bridge_state_name == "TOP":
                if not top_slow_active:
                    print("[BRIDGE] IMU=TOP & MID -> TOP slow ACTIVE")
                top_slow_active = True

            # ================= TRAFFIC LIGHT BOX PICK =================
            best_tlight_box = None
            best_tlight_conf = 0.0
            if tlight_candidates:
                best_tlight_box, best_tlight_conf, _ = max(
                    tlight_candidates,
                    key=lambda t: (t[1] >= TLIGHT_DET_CONF_HIGH, t[1], t[2])
                )

            # ================= SIGN UPDATE =================
            sign_state.step()
            sign_state.update_from_dets(sign_labels, sign_confs)
            cur_sign = sign_state.current()

            sign_mem.update(cur_sign)
            sign_mem.step()
            turn_state.step()
            sign_for_turn = sign_mem.get() if sign_mem.get() is not None else cur_sign

            # ================= TRAFFIC LIGHT COLOR + STABLE =================
            tlight_color = None
            if best_tlight_box is not None:
                tlight_color = classify_traffic_light_color_robust(img3, best_tlight_box)

            tlight_state.update(tlight_color)
            cur_tlight = tlight_state.current()

            if cur_tlight != prev_tl:
                print(f"[TLIGHT] {prev_tl} -> {cur_tlight}")
                if prev_tl == 'red' and cur_tlight == 'green':
                    green_delay_left = TLIGHT_GREEN_GO_DELAY_FRAMES
                    green_protect_left = TLIGHT_AFTER_GREEN_IGNORE_RED_FRAMES
                elif cur_tlight == 'green' and prev_tl != 'green':
                    green_protect_left = TLIGHT_AFTER_GREEN_IGNORE_RED_FRAMES
                if cur_tlight == 'red':
                    green_delay_left = 0
                    green_protect_left = 0
                prev_tl = cur_tlight

            if green_protect_left > 0:
                green_protect_left -= 1

            # ================= LANE FILTER =================
            lane_mask = smooth_lane_edges(lane_mask, erode_iter=1, dilate_iter=1, kernel_size=3)
            lane_mask_full = lane_mask.copy()  # keep branches for intersection detect

            lane_follow = keep_largest_lane(lane_mask, min_area=300)
            lane_follow = apply_trapezoid_roi(lane_follow)

            lane_mask_full = lane_mask_full  # full (no trapezoid) for intersection
            lane_mask = lane_follow

            lane_sum = int(lane_mask.sum())
            h_mask, w_mask = lane_mask.shape
            cx = w_mask // 2

            path_cx = near_cx = far_cx = None
            if lane_sum > 0:
                path_cx, near_cx, far_cx = compute_path_center(lane_mask)

            if path_cx is None and road_boxes:
                mid_box = get_lane_from_road_boxes(road_boxes)
                if mid_box is not None:
                    path_cx = mid_box
                    near_cx = mid_box
                    far_cx  = mid_box

            curvature = 0.0
            if far_cx is not None and near_cx is not None:
                curvature = abs(far_cx - near_cx)

            lane_found = (path_cx is not None)

            # ================= DRIVE LOGIC =================
            speed_cmd = 0
            angle_cmd = 0.0
            is_intersection = False

            if lane_found:
                if mode != MODE_FORWARD:
                    pid.reset()
                    mode = MODE_FORWARD
                lost_frames = 0

                error_px = path_cx - cx
                base_speed_eff = BASE_SPEED

                if curvature > CURV_THRESH_2:
                    # On DOWN: don't boost error too much (more stable)
                    if not (bridge_armed and bridge_state_name == "DOWN"):
                        error_px *= CURV_ERROR_BOOST_2
                    base_speed_eff = BASE_SPEED - 8
                elif curvature > CURV_THRESH_1:
                    if not (bridge_armed and bridge_state_name == "DOWN"):
                        error_px *= CURV_ERROR_BOOST_1
                    base_speed_eff = BASE_SPEED - 4

                # Intersection + Turn commit
                is_intersection = detect_intersection(lane_mask_full)
                sign_for_turn = sign_mem.get() if sign_mem.get() is not None else cur_sign

                if (not turn_state.is_active()) and is_intersection and (sign_for_turn is not None):
                    man = decide_maneuver_from_sign(sign_for_turn, lane_mask_full)
                    if man is not None:
                        targets = branch_targets_from_mask(lane_mask_full)
                        target_cx = None if targets is None else targets.get(man, None)
                        turn_state.start(man, w_mask, target_cx=target_cx)
                        sign_mem.consume()

                # Error command
                if turn_state.is_active():
                    error_cmd = turn_state.apply(error_px)
                    if turn_state.current() in ('left', 'right'):
                        base_speed_eff = min(base_speed_eff, TURN_SPEED)
                        min_spd = TURN_SPEED
                    else:
                        min_spd = MIN_SPEED
                else:
                    error_cmd = apply_sign_logic(error_px, lane_mask, cur_sign, cx)
                    min_spd = MIN_SPEED

                angle_cmd = -pid.compute(error_cmd)
                angle_cmd = max(min(angle_cmd, MAX_ANGLE), -MAX_ANGLE)

                # Bridge DOWN: reduce angle
                if bridge_armed and bridge_state_name == "DOWN":
                    angle_cmd *= BRIDGE_DOWN_ANGLE_COEFF
                    angle_cmd = max(min(angle_cmd, BRIDGE_DOWN_MAX_ANGLE), -BRIDGE_DOWN_MAX_ANGLE)

                last_good_angle_cmd = angle_cmd
                angle_history.append(angle_cmd)

                # Base speed
                if sign_for_turn == 'stop_sign' or cur_sign == 'stop_sign':
                    speed_cmd = 0
                else:
                    speed_cmd = adaptive_speed_control(error_cmd, base_speed=base_speed_eff, min_speed=min_spd)

                # ================= BRIDGE SPEED POLICY =================
                top_policy_ok  = bridge_armed and bridge2_seen
                down_policy_ok = bridge_armed and (bridge2_seen or down_frames >= BRIDGE_DOWN_FALLBACK_FR)

                if bridge_armed:
                    if bridge_state_name == "TOP":
                        if top_policy_ok and top_slow_active:
                            speed_cmd = int(speed_cmd * BRIDGE_TOP_SLOW_COEFF)
                        else:
                            speed_cmd = int(speed_cmd * bridge_speed_coeff)

                    elif bridge_state_name == "DOWN":
                        if down_policy_ok:
                            if (not down_stop1_done) and (down_frames >= BRIDGE_DOWN_STOP1_AT_FR):
                                down_stop_frames_left = BRIDGE_DOWN_STOP_HOLD_FR
                                down_stop1_done = True
                                print("[BRIDGE-DOWN] Stop #1 mid-slope")

                            if (not down_stop2_done) and \
                               (down_frames >= BRIDGE_DOWN_MIN_FR_BEFORE_END) and \
                               (abs(pitch_filtered) <= BRIDGE_DOWN_END_PITCH_DEG):
                                down_stop_frames_left = BRIDGE_DOWN_STOP_HOLD_FR
                                down_stop2_done = True
                                print("[BRIDGE-DOWN] Stop #2 near-end")

                            if down_stop_frames_left > 0:
                                down_stop_frames_left -= 1
                                speed_cmd = 0
                            else:
                                speed_cmd = int(speed_cmd * BRIDGE_DOWN_SLOW_COEFF)
                        else:
                            speed_cmd = int(speed_cmd * bridge_speed_coeff)

                    elif bridge_state_name == "UP":
                        speed_cmd = int(speed_cmd * bridge_speed_coeff)
                    else:
                        speed_cmd = int(speed_cmd * bridge_speed_coeff)
                else:
                    speed_cmd = int(speed_cmd * bridge_speed_coeff)

                if speed_cmd < 0:
                    speed_cmd = 0

            else:
                lost_frames += 1
                if mode == MODE_FORWARD:
                    if lost_frames == 1:
                        speed_cmd = 0
                        angle_cmd = 0.0
                    elif lost_frames >= RECOV_LOST_FRAMES_TRIGGER:
                        if len(angle_history) > 0:
                            base_turn_angle = max(angle_history, key=lambda a: abs(a))
                        else:
                            base_turn_angle = last_good_angle_cmd
                        base_turn_angle = max(min(base_turn_angle, MAX_ANGLE), -MAX_ANGLE)
                        mode = MODE_RECOV_BACK
                        speed_cmd = RECOV_REVERSE_SPEED
                        angle_cmd = -base_turn_angle
                else:
                    speed_cmd = RECOV_REVERSE_SPEED
                    angle_cmd = -base_turn_angle

            # ================= TLIGHT APPROACH SLOW =================
            if best_tlight_box is not None and cur_tlight is None:
                speed_cmd = min(speed_cmd, TLIGHT_APPROACH_SPEED)
            if tlight_color == 'red' and cur_tlight is None:
                speed_cmd = min(speed_cmd, TLIGHT_APPROACH_SPEED)

            # ================= TLIGHT OVERRIDE (TOP PRIORITY) =================
            if cur_tlight == 'red':
                speed_cmd = 0
                angle_cmd = 0.0
                green_delay_left = 0
                green_protect_left = 0
            elif green_delay_left > 0:
                speed_cmd = 0
                angle_cmd = 0.0
                green_delay_left -= 1

            # ================= SEND CONTROL =================
            control(speed_cmd, angle_cmd)

            # ================= LOG =================
            print(
                f"[MODEL]={active_model_name} bridge_active={int(bridge_active)} "
                f"bridge_state={bridge_state_name} armed={int(bridge_armed)} mid={int(bridge2_seen)} "
                f"pitch={pitch_filtered:.1f} downFr={down_frames} stopLeft={down_stop_frames_left} | "
                f"sign={cur_sign} mem={sign_mem.get()} man={turn_state.current()} inter={int(is_intersection)} | "
                f"tl_conf={best_tlight_conf:.2f} tl_raw={tlight_color} tl={cur_tlight} green_wait={green_delay_left} | "
                f"lane={int(lane_found)} lost={lost_frames} mode={mode} speed={speed_cmd} angle={angle_cmd:.1f}"
            )

            # ================= VIS + RECORD =================
            t1 = time.perf_counter()
            fps = 1.0 / max(t1 - t0, 1e-6)
            fps_hist.append(fps)
            frame_count += 1
            avg_fps = sum(fps_hist) / len(fps_hist)

            colorlist = gen_color(len(CLASSES))
            if isinstance(masks, np.ndarray) and isinstance(boxes, np.ndarray) and boxes.size > 0:
                _, vis_img = vis_result(img3, (boxes, masks, (H, W)), colorlist, CLASSES, onScreen=True)
            else:
                vis_small = cv2.resize(frame, (INPUT_W, INPUT_H))
                vis_img = np.concatenate([vis_small, vis_small, vis_small], axis=1)

            overlay = (
                f'FPS:{avg_fps:.1f} model:{active_model_name} '
                f'sign:{cur_sign} man:{turn_state.current()} inter:{int(is_intersection)} '
                f'tl:{cur_tlight} wait:{green_delay_left} '
                f'bridge:{bridge_state_name} mid:{int(bridge2_seen)} pitch:{pitch_filtered:.1f} '
                f'lost:{lost_frames} speed:{speed_cmd} angle:{angle_cmd:.1f}'
            )
            cv2.putText(vis_img, overlay, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(vis_img)

            if lost_frames >= MAX_LOST_FRAMES:
                print(f"[WARN] lost_frames={lost_frames} (>= {MAX_LOST_FRAMES})")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        print("[FINAL] Send (0,0) and release.")
        control(0, 0)
        try:
            cap.release()
        except Exception:
            pass
        try:
            rknn_v11.release()
        except Exception:
            pass
        try:
            rknn_v9.release()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
                print("Saved video:", video_path)
            except Exception:
                pass
        close_serial()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
