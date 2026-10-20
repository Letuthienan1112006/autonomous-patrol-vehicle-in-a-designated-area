
import cv2
import numpy as np
import time
from ultralytics import YOLO
import road
import threading
import record

model= YOLO("yolov8s.pt")

cap=cv2.VideoCapture(0)
last_alert_ts=0
alert_cooldown=8  # seconds
while True:
    ret,frame=cap.read()
    if not ret: break
    frame=cv2.resize(frame,(640,480))

    off,mask=road.lane_offset(frame,roi_top=300,white=True)

    res=model(frame, imgsz=640, conf=0.6, verbose=False)[0]
    best_conf=0.0
    for b in res.boxes:
        if int(b.cls)==0:  # class 0=person
            c=float(b.conf); best_conf=max(best_conf,c)
            x1,y1,x2,y2=map(int,b.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"person {c:.2f}",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    now=time.time()
    if best_conf>0.7 and now-last_alert_ts>alert_cooldown:
        cv2.imwrite('snap.jpg',frame)
        threading.Thread(target=record.record_clip, args=(cap,6,"clip.mp4",20), daemon=True).start()
        print(f"[ALERT] Person detected conf={best_conf:.2f}")
        last_alert_ts=now

    cv2.putText(frame,f"offset:{0 if off is None else int(off)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("AGV View", frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release()
cv2.destroyAllWindows()        

       

        