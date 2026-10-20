import cv2

# 0 = laptop cam, 2 = C270 (theo v4l2-ctl của bạn)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Không mở được C270, thử lại với index khác (1, 2, 3...)")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame")
        break

    cv2.imshow("C270 HD Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
