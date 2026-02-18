# server.py
import os, zmq, msgpack, msgpack_numpy as m, numpy as np, cv2
from ultralytics import YOLO
m.patch()

PORT  = 5555
MODEL = "yolo11n.pt"
# MODEL = os.getenv("YOLO_MODEL_PATH", "./yolo11n_ncnn_model")

CONF  = float(os.getenv("CONF", "0.35"))
IMG   = int(os.getenv("IMG", "416"))

def main():
    print(f"Obnav split-compute server on 0.0.0.0:{PORT}  model={MODEL}")
    model = YOLO(MODEL, task="detect")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{PORT}")

    while True:
        req = sock.recv()
        data = msgpack.unpackb(req, raw=False)

        # decode JPEG
        jpg = np.frombuffer(data["img"], dtype=np.uint8)
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

        # run YOLO
        r = model.predict(source=frame, imgsz=IMG, conf=CONF, verbose=False)[0]
        boxes = []
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), c, k in zip(xyxy, confs, clss):
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(c), int(k)])
                # draw box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                label = model.names[int(k)] if hasattr(model, "names") else str(k)
                cv2.putText(frame, f"{label} {c:.2f}", (x1, max(15,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # reply
        sock.send(msgpack.packb({"boxes": boxes}, use_bin_type=True))
        print(f"infer result: {len(boxes)} boxes")

        # show received stream with detections
        cv2.imshow("Server Stream", frame)
        if cv2.waitKey(1) == 27:  # Esc to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()