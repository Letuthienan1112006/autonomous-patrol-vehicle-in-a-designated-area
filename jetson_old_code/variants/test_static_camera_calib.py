"""
BUOC 2 - Test tinh hieu chinh camera, KHONG can chay xe.

Dat xe DUNG YEN, canh camera cho THAT SU chinh giua vach vang that
(do bang thuoc/day, khong doan bang mat), roi chay file nay. Neu Err
bao ve gan 0 (+-10px) thi camera dang hieu chuan dung. Neu Err lech
xa 0 dang ke du da canh giua that su bang thuoc, do la bang chung
ro rang cho loi hieu chinh camera/hinh hoc (STEER_CENTER_X=320 khong
khop voi tam xe that), khong phai loi dong/PID/FPS.

CACH DUNG:
  1. Dat xe dung yen tren mat dat that.
  2. Do dac chinh xac: tam camera (nhin theo huong thang) phai trung
     voi vach vang o phia truoc, dung thuoc/day de canh, khong uoc luong.
  3. Chay: python3 test_static_camera_calib.py
  4. Doc gia tri Err in ra. Lap lai o 2-3 vi tri/doan duong khac nhau
     (van canh giua chinh xac moi lan) de xem Err co nhat quan gan 0
     khong, hay luon lech ve 1 huong co dinh (bang chung manh hon cho
     loi camera thay vi nhieu ngau nhien).
"""
import sys
import time

sys.path.insert(0, "/home/agv/agv_git_clean")

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from Confg import (
    SEGMENTATION_MODEL_PATH, SEG_NUM_CLASSES, SEG_BACKBONE,
    SEG_INPUT_SIZE, SEG_NORMALIZE_MEAN, SEG_NORMALIZE_STD,
    CLASS_COLORS, SEG_CLASS_NAMES, STEER_CENTER_X,
)
from model import create_deeplabv3


def get_lane_midpoint(mask, y):
    """Sao chep dung logic tu patrol_robot.py de dam bao nhat quan."""
    line_indices = np.where(mask[y, :] == 2)[0]
    if len(line_indices) >= 2:
        left = line_indices[0]
        right = line_indices[-1]
        return left, right, (left + right) // 2
    lane_indices = np.where(mask[y, :] == 1)[0]
    if len(lane_indices) < 2:
        return None
    left = lane_indices[0]
    right = lane_indices[-1]
    return left, right, (left + right) // 2


def main():
    print("=== Test tinh hieu chinh camera ===")
    print(f"STEER_CENTER_X hien tai = {STEER_CENTER_X}")
    input("Dam bao camera da canh CHINH GIUA vach vang (do bang thuoc). "
          "Nhan Enter de chup...")

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    frame = None
    for _ in range(25):
        ret, frame = cap.read()
        time.sleep(0.05)
    cap.release()
    cv2.imwrite("/tmp/static_calib_raw.jpg", frame)

    bgr = frame
    pil_im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    img_transform = transforms.Compose([
        transforms.Resize(SEG_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(SEG_NORMALIZE_MEAN, SEG_NORMALIZE_STD),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_deeplabv3(
        num_classes=SEG_NUM_CLASSES, backbone=SEG_BACKBONE, pretrained=False
    ).to(device)
    ckpt = torch.load(SEGMENTATION_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    use_fp16 = device.type == "cuda"
    if use_fp16:
        model = model.half()

    tensor = img_transform(pil_im).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()

    with torch.no_grad():
        out = model(tensor)
        pred_small = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

    frame_h, frame_w = bgr.shape[:2]
    pred_mask = cv2.resize(
        pred_small.astype(np.uint8), (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    )
    # Ap dung fix motobike->road giong patrol_robot.py de nhat quan
    pred_mask[pred_mask == 4] = 1

    scan_y = int(frame_h * 0.85)  # gan giong INTERSECTION_SCAN_Y thuc te
    result = get_lane_midpoint(pred_mask, scan_y)

    color_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        color_mask[pred_mask == cid] = color
    blend = cv2.addWeighted(bgr, 0.5, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR), 0.5, 0)
    cv2.line(blend, (STEER_CENTER_X, 0), (STEER_CENTER_X, frame_h), (255, 255, 255), 1)
    cv2.line(blend, (0, scan_y), (frame_w, scan_y), (255, 0, 255), 1)

    if result is None:
        print("\n[LOI] Khong tim thay road/line tai hang scan - kiem tra lai vi tri xe.")
    else:
        left, right, mx = result
        err = int(mx) - STEER_CENTER_X
        cv2.circle(blend, (int(mx), scan_y), 5, (0, 255, 255), -1)
        print(f"\nleft={left}  right={right}  midpoint={mx}")
        print(f"STEER_CENTER_X={STEER_CENTER_X}")
        print(f">>> Err = midpoint - STEER_CENTER_X = {err} <<<")
        if abs(err) <= 10:
            print("=> Gan 0, camera co ve hieu chuan dung tai vi tri nay.")
        else:
            print("=> LECH DANG KE du da canh giua bang thuoc - nghi ngo loi "
                  "hieu chinh camera/hinh hoc.")

    cv2.imwrite("/tmp/static_calib_overlay.jpg", blend)
    print("\nDa luu: /tmp/static_calib_raw.jpg /tmp/static_calib_overlay.jpg")


if __name__ == "__main__":
    main()
