import cv2
import time





def record_clip(cam, secs=6, out="clip.mp4", fps=20):

    w  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    h  = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    t0 = time.time()
    while time.time()-t0 < secs:
        ret, fr = cam.read()
        if not ret: break
        wr.write(fr)
    wr.release()