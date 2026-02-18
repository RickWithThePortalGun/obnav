from __future__ import annotations
import csv, time, os, logging
from typing import Optional, Dict, Any

class EvalLogger:
    def __init__(self, path: str = "eval_log.csv", flush_every: int = 30, detector_type: Optional[str] = None):
        """
        detector_type: Optional; pass "remote" or "local" when initializing
        """
        self.path = path
        self.flush_every = flush_every
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
        if os.stat(self.path).st_size == 0:
            self._w.writerow([
                "ts", "fps", "latency_ms", "num_dets", "action", "path",
                "left_cm", "front_cm", "right_cm", "detector",
                "ultrasonic_cm", "vision_cm"
            ])

        self._last_flush = time.time()
        self._last_tick = None
        self.detector_type = detector_type or os.getenv("DETECTOR_MODE", "unknown")

        # setup a lightweight logger
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.INFO,
        )
        logging.info(f"EvalLogger initialized — detector in use: {self.detector_type}")

    def tick_start(self):
        self._t0 = time.time()

    def tick_end(self, info: Dict[str, Any], action: Optional[str], path: Optional[str]):
        t1 = time.time()
        lat_ms = (t1 - getattr(self, "_t0", t1)) * 1000.0
        if self._last_tick is None:
            fps = 0.0
        else:
            dt = t1 - self._last_tick
            fps = 1.0 / dt if dt > 0 else 0.0
        self._last_tick = t1

        sd = info.get("sector_dist", {})
        num_dets = len(info.get("dets", []))

        row = [
            int(t1),
            round(fps, 2),
            round(lat_ms, 1),
            num_dets,
            action or "",
            path or "",
            int(sd.get("left", 999)),
            int(sd.get("front", 999)),
            int(sd.get("right", 999)),
            self.detector_type,
            round(info.get("ultrasonic_cm", -1), 1),
            round(info.get("vision_cm", -1), 1)
        ]

        self._w.writerow(row)
        logging.info(
            f"[{self.detector_type.upper()}] "
            f"FPS={fps:.2f}, Lat={lat_ms:.1f}ms, Dets={num_dets}, Action={action or '-'}"
        )

        if (t1 - self._last_flush) > self.flush_every:
            self._f.flush()
            self._last_flush = t1
            logging.info("Flushed eval log to disk.")

    def close(self):
        try:
            self._f.flush()
            self._f.close()
            logging.info("EvalLogger closed cleanly.")
        except Exception as e:
            logging.warning(f"EvalLogger close failed: {e}")
