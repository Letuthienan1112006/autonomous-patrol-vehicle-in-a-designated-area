from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")  
cap = cv2.VideoCapture(0)






while True:
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, imgsz=640, conf=0.5, verbose=False)[0]

   
   
   
   
    for b in res.boxes:
        if int(b.cls) == 0:  # class 0 = person,1=bicycle,2=car
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            conf = float(b.conf)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"person {conf:.2f}", (x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

   
   
   
   
   
   
   
   
   
   
    cv2.imshow("YOLO person", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release(); cv2.destroyAllWindows()