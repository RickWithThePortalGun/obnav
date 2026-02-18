# client_smoke.py
import cv2, zmq, msgpack, numpy as np
cap=cv2.VideoCapture(0); _,f=cap.read(); cap.release()
_,jpg=cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY),60])
ctx=zmq.Context(); s=ctx.socket(zmq.REQ); s.connect("tcp://0.0.0.0:5555")
s.send(msgpack.packb({"img": jpg.tobytes(), "conf": 0.35}, use_bin_type=True))
cv2.imshow("sent", f)
cv2.waitKey(0)
resp = msgpack.unpackb(s.recv(), raw=False)
print(resp["boxes"])  # [x1,y1,x2,y2,conf,cls]
