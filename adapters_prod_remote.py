import os

from ultralytics import YOLO
import zmq
import msgpack
import cv2
import numpy as np
import time
import random
from typing import List
from ports import Detector, Box

class RemoteDetector(Detector):
    REQUEST_TIMEOUT = 2000
    REQUEST_RETRIES = 3
    BACKOFF_MAX = 4
    def __init__(self, server,
                 port=5555,
                 conf=0.30, q=60, w=320, h=320):
        self.addr = f"tcp://{server}:{port}"
        self.conf = conf
        self.q = q
        self.w = w
        self.h = h
        self.context = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.RCVTIMEO, self.REQUEST_TIMEOUT)
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.connect(self.addr)

    def infer(self, frame: np.ndarray) -> List[Box]:
        H, W, _ = frame.shape
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        sq = frame[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(sq, (self.w, self.h))
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.q])
        if not ok:
            return []
        payload = {"img": jpg.tobytes(), "w": self.w, "h": self.h, "conf": self.conf}

        retries_left = self.REQUEST_RETRIES
        while retries_left:
            try:
                self.client.send(msgpack.packb(payload, use_bin_type=True))
                poller = zmq.Poller()
                poller.register(self.client, zmq.POLLIN)
                if poller.poll(self.REQUEST_TIMEOUT):
                    resp = msgpack.unpackb(self.client.recv(), raw=False)
                    return [tuple(b) for b in resp.get("boxes", [])]
                else:
                    retries_left -= 1
                    if retries_left == 0:
                        break
                    self.client.setsockopt(zmq.LINGER, 0)
                    self.client.close()
                    poller.unregister(self.client)
                    time.sleep(0.1 * (2 ** random.randint(0, self.BACKOFF_MAX)))
                    self._connect()
            except zmq.ZMQError:
                retries_left -= 1
                if retries_left == 0:
                    break
                # Reconnect as above
                self.client.setsockopt(zmq.LINGER, 0)
                self.client.close()
                time.sleep(0.1 * (2 ** random.randint(0, self.BACKOFF_MAX)))
                self._connect()
        print("E: Remote server unavailable after retries; consider fallback")
        return []

    def test_connection(self) -> bool:
        """Send a dummy request to test connectivity."""
        print(f"Testing connection to {self.addr}")
        dummy_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        try:
            result = self.infer(dummy_frame)
            print(f"Connection test result: {len(result)} detections")
            # Connection is successful if we get a response (even if 0 detections)
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

class LocalDetector(Detector):
    model_path = os.getenv("YOLO_MODEL_PATH", "./yolo11n_ncnn_model")
    def __init__(self, model_path=model_path, conf=0.30):
        self.model = YOLO(model_path)
        self.conf = conf
    def infer(self, frame: np.ndarray) -> List[Box]:
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes = []
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            boxes.append((x1, y1, x2, y2, conf, cls))
        return boxes