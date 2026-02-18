from __future__ import annotations

import os
from typing import List, Dict, Optional

from ports import Detector, Camera, Rangefinder, Haptics, Box
import numpy as np
import time
from collections import deque

INVALID_DIST: float = 999.0

CONF_THRESH: float = float(os.getenv("CONF_THRESH", "0.35"))
DIST_CLOSE_CM: float = float(os.getenv("DIST_CLOSE_CM", "80"))
OBSTACLE_CLASSES: set[int] = {int(x) for x in os.getenv("OBSTACLE_CLASSES", "0,56,60").split(",")}
LEFT_SPLIT: float = float(os.getenv("LEFT_SPLIT", "0.30"))
RIGHT_SPLIT: float = float(os.getenv("RIGHT_SPLIT", "0.70"))
FOCAL_PX: float = float(os.getenv("FOCAL_PX", "529"))

REAL_H_CM: Dict[int, float] = {
    0: 165,
    56: 50,
    57: 85,
    60: 75,
}
REAL_H_DICT: Dict[int, float] = {
    cls: float(os.getenv(f"REAL_H_CM_{cls}", REAL_H_CM.get(cls, 150)))
    for cls in OBSTACLE_CLASSES
}

SECTORS = ("left", "front", "right")




def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    areaA = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    areaB = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union = areaA + areaB - inter
    return 0.0 if union <= 0 else (inter / union)

def bbox_from_state(cx, cy, w_px, h_px):
    x1 = cx - 0.5 * w_px
    y1 = cy - 0.5 * h_px
    x2 = cx + 0.5 * w_px
    y2 = cy + 0.5 * h_px
    return (x1, y1, x2, y2)


class KFTrack:
    """
    Constant Velocity Kalman Filter for detection tracks.
    State: [cx, cy, vx, vy, z, vz] (cx,cy in pixels, z in cm)
    Measurements: [cx, cy, z] (z may be None)
    """
    def __init__(self, initial_cx, initial_cy, initial_z, bbox_wh=(60,180), track_id=0, now_ts=None):
        # state vector
        self.x = np.array([
            initial_cx,            # cx
            initial_cy,            # cy
            0.0,                   # vx
            0.0,                   # vy
            initial_z if initial_z is not None else 200.0,  # z (cm), fallback 200 cm
            0.0                    # vz
        ], dtype=float)

        self.P = np.diag([50.0, 50.0, 25.0, 25.0, 500.0, 50.0])  # tune as needed

        q_pos = 1.0
        q_vel = 5.0
        q_z = 50.0
        q_zv = 10.0
        self.Q_base = np.diag([q_pos, q_pos, q_vel, q_vel, q_z, q_zv])

        self.R_base = np.diag([25.0, 25.0, 400.0])  # pixel noise for cx/cy, cm^2 for z

        self.last_update = now_ts if now_ts is not None else time.time()
        self.last_seen = self.last_update
        self.id = track_id
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.bbox_wh = bbox_wh  # default width/height in pixels for reconstruct bbox
        self.history = deque(maxlen=20)

    def predict(self, dt):
        """
        Predict step of linear KF with dt (seconds).
        """
        if dt <= 0:
            self.age += 1
            return

        F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Q scaled by dt (simple approximation)
        Q = self.Q_base * max(1.0, dt)

        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.age += 1
        # store history
        self.history.append((time.time(), self.x.copy()))

    def update(self, meas_cx=None, meas_cy=None, meas_z=None):
        """
        Update step. Any of meas_cx/meas_cy/meas_z may be None -> skip those dimensions.
        """
        # Build measurement vector and H, R dynamically.
        zs = []
        H_rows = []
        R_rows = []

        if meas_cx is not None:
            zs.append(meas_cx)
            H_rows.append([1, 0, 0, 0, 0, 0])  # cx
            R_rows.append([self.R_base[0,0]])
        if meas_cy is not None:
            zs.append(meas_cy)
            H_rows.append([0, 1, 0, 0, 0, 0])  # cy
            R_rows.append([self.R_base[1,1]])
        if meas_z is not None:
            zs.append(meas_z)
            H_rows.append([0, 0, 0, 0, 1, 0])  # z
            R_rows.append([self.R_base[2,2]])

        if len(zs) == 0:
            # nothing to update
            self.misses += 1
            return

        z = np.array(zs, dtype=float).reshape((-1,1))
        H = np.array(H_rows, dtype=float).reshape((len(zs), 6))
        R = np.diag([r[0] for r in R_rows])

        # KF standard equations (Kalman gain form)
        x_prior = self.x.reshape((6,1))
        P_prior = self.P

        S = H @ P_prior @ H.T + R  # innovation covariance
        # numerical safety
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_prior @ H.T @ S_inv
        y = z - (H @ x_prior)  # innovation
        x_post = x_prior + K @ y
        I = np.eye(6)
        P_post = (I - K @ H) @ P_prior

        self.x = x_post.ravel()
        self.P = P_post
        self.last_update = time.time()
        self.last_seen = self.last_update
        self.hits += 1
        self.misses = 0

    def get_state(self):
        return self.x.copy()

    def get_centroid(self):
        return float(self.x[0]), float(self.x[1])

    def get_depth(self):
        return float(self.x[4])

    def as_bbox(self):
        cx, cy = self.get_centroid()
        w_px, h_px = self.bbox_wh
        return bbox_from_state(cx, cy, w_px, h_px)

class Tracker:
    def __init__(self,
                 max_age=0.8,
                 min_hits=1,
                 iou_threshold=0.15,
                 default_bbox_wh=(60,180)):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.default_bbox_wh = default_bbox_wh
        self.last_ts = time.time()

    def predict_all(self, now_ts=None):
        if now_ts is None:
            now_ts = time.time()
        dt = now_ts - self.last_ts
        # cap dt for numerical stability
        dt = max(1e-3, min(1.0, dt))
        for t in self.tracks:
            t.predict(dt)
        self.last_ts = now_ts

    def _match_detections(self, detections):
        """
        detections: list of detections in format
          [{'bbox':(x1,y1,x2,y2), 'cx':cx, 'cy':cy, 'z':z_est, 'conf':conf, 'cls':cls}, ...]
        returns matched_pairs: list of (track_idx, det_idx), unmatched_tracks, unmatched_dets
        Uses greedy IoU matching (descending).
        """
        N = len(self.tracks)
        M = len(detections)
        if N == 0:
            return [], list(range(N)), list(range(M))

        # Build IoU matrix (tracks x detections)
        iou_mat = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            tr_bbox = tr.as_bbox()
            for j, det in enumerate(detections):
                iou_mat[i, j] = iou(tr_bbox, det['bbox'])

        # Greedy matching: pick the highest IoU pairs until none above threshold
        pairs = []
        matched_tracks = set()
        matched_dets = set()
        # flatten with indices
        ij = [(iou_mat[i, j], i, j) for i in range(N) for j in range(M)]
        ij.sort(key=lambda x: x[0], reverse=True)
        for score, i, j in ij:
            if score < self.iou_threshold:
                break
            if i in matched_tracks or j in matched_dets:
                continue
            matched_tracks.add(i)
            matched_dets.add(j)
            pairs.append((i, j))

        unmatched_tracks = [i for i in range(N) if i not in matched_tracks]
        unmatched_dets = [j for j in range(M) if j not in matched_dets]
        return pairs, unmatched_tracks, unmatched_dets

    def update(self, detections, now_ts=None):
        """
        detections: list of dicts: {'bbox':(x1,y1,x2,y2), 'conf':, 'cls':, 'cx':, 'cy':, 'z':}
        """
        if now_ts is None:
            now_ts = time.time()

        # 1) Predict all tracks to now
        self.predict_all(now_ts)

        # 2) Match detections to tracks
        pairs, unmatched_tracks, unmatched_dets = self._match_detections(detections)

        # 3) Update matched tracks with measurement
        for track_idx, det_idx in pairs:
            tr = self.tracks[track_idx]
            det = detections[det_idx]
            tr.update(meas_cx=det.get('cx'), meas_cy=det.get('cy'), meas_z=det.get('z'))
            tr.bbox_wh = (det.get('w', tr.bbox_wh[0]), det.get('h', tr.bbox_wh[1]))
            tr.last_seen = now_ts

        # 4) Create new tracks for unmatched detections
        for j in unmatched_dets:
            det = detections[j]
            init_z = det.get('z')
            cx = det.get('cx'); cy = det.get('cy')
            t = KFTrack(initial_cx=cx, initial_cy=cy, initial_z=init_z,
                        bbox_wh=(det.get('w', self.default_bbox_wh[0]),
                                 det.get('h', self.default_bbox_wh[1])),
                        track_id=self.next_id, now_ts=now_ts)
            self.next_id += 1
            self.tracks.append(t)

        # 5) Increase miss counters for unmatched tracks
        for i in unmatched_tracks:
            tr = self.tracks[i]
            tr.misses += 1

        # 6) Prune dead tracks and distant tracks
        keep = []
        for tr in self.tracks:
            age_since_seen = now_ts - tr.last_seen
            # Remove if too old or too far away
            if age_since_seen > self.max_age:
                continue
            # Remove tracks that are too far (over 5 meters)
            if tr.x[4] > 500:  # z > 500cm
                continue
            # Remove tracks with too many misses relative to hits
            if tr.misses > 3 and tr.hits < 2:
                continue
            keep.append(tr)
        self.tracks = keep

    def get_active_tracks(self):
        """
        Returns list of dictionaries for active tracks:
         {'id', 'cx', 'cy', 'z', 'bbox', 'age', 'hits', 'misses'}
        """
        out = []
        for tr in self.tracks:
            bx = tr.as_bbox()
            out.append({
                'id': tr.id,
                'cx': float(tr.x[0]),
                'cy': float(tr.x[1]),
                'z': float(tr.x[4]),
                'bbox': (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                'age': tr.age,
                'hits': tr.hits,
                'misses': tr.misses
            })
        return out


def sector_from_box(x1: float, x2: float, left_px: float, right_px: float) -> str:
    cx = 0.5 * (x1 + x2)
    if cx < left_px:
        return "left"
    if cx > right_px:
        return "right"
    return "front"


def estimate_distance_cm(y1: float, y2: float, cls: int, focal_px: float) -> Optional[float]:
    h = y2 - y1
    if h <= 1:
        return None
    return (focal_px * REAL_H_DICT.get(cls, 150)) / h


tracker=Tracker(max_age=0.5, min_hits=1, iou_threshold=0.25, default_bbox_wh=(60, 180))
def step_once(frame, detections, ultrasonic_cm=None, ts=None):
    """
    frame: image frame if needed (unused here)
    detections: list of detections from server in format
       [x1,y1,x2,y2,conf,cls] or similar
    ultrasonic_cm: a single ultrasonic reading for frontal distance (optional)
    ts: timestamp in seconds (optional)
    """
    if ts is None:
        ts = time.time()

    det_list = []
    vision_depths = []
    H_img, W_img = frame.shape[0], frame.shape[1]
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        vision_z = estimate_distance_cm(y1, y2, cls, focal_px=FOCAL_PX)
        z_cm = vision_z
        
        if vision_z is not None:
            vision_depths.append(vision_z)

        det_list.append({
            'bbox': (x1, y1, x2, y2),
            'conf': conf,
            'cls': cls,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'z': z_cm,
            'vision_z': vision_z
        })

    tracker.update(det_list, now_ts=ts)
    tracks = tracker.get_active_tracks()
    sectors = {'left': None, 'front': None, 'right': None}
    left_cut = int(W_img * LEFT_SPLIT)
    right_cut = int(W_img * RIGHT_SPLIT)

    for tr in tracks:
        x1, y1, x2, y2 = tr['bbox']
        cx = tr['cx']
        z = tr['z']
        
        # Apply confidence decay based on age and misses
        age_factor = max(0.3, 1.0 - (tr['age'] * 0.1))  # Decay with age
        miss_factor = max(0.5, 1.0 - (tr['misses'] * 0.2))  # Decay with misses
        confidence = age_factor * miss_factor
        
        # Skip low-confidence tracks
        if confidence < 0.4:
            continue

        # Decide which sector this belongs to
        if cx < left_cut:
            s = 'left'
        elif cx > right_cut:
            s = 'right'
        else:
            s = 'front'

        # Use ultrasonic sensor for all sectors since we only have one
        if ultrasonic_cm is not None and 5 < ultrasonic_cm < 400:
            if s == 'front':
                # Front sector: blend vision and ultrasonic (ultrasonic is more reliable)
                if z is not None:
                    z = 0.3 * z + 0.7 * ultrasonic_cm
                else:
                    z = ultrasonic_cm
            else:
                # Left/Right sectors: use ultrasonic as reference, adjust vision estimate
                if z is not None:
                    # Scale vision estimate based on ultrasonic reading
                    # If vision says close but ultrasonic says far, trust ultrasonic more
                    if z < ultrasonic_cm * 0.8:  # Vision much closer than ultrasonic
                        z = ultrasonic_cm * 0.9  # Use ultrasonic as upper bound
                    else:
                        z = 0.6 * z + 0.4 * ultrasonic_cm  # Blend estimates
                else:
                    # No vision estimate, use ultrasonic with sector adjustment
                    if s == 'left':
                        z = ultrasonic_cm * 1.1  # Slightly further for left
                    elif s == 'right':
                        z = ultrasonic_cm * 1.1  # Slightly further for right
                    else:
                        z = ultrasonic_cm

        if z is None:
            continue

        # Apply confidence to distance (closer tracks with higher confidence win)
        effective_z = z / confidence
        
        if sectors[s] is None or effective_z < sectors[s]['z']:
            sectors[s] = {'z': z, 'track': tr, 'confidence': confidence}
    
    # Fallback: If no vision detections but ultrasonic data available, use ultrasonic for all sectors
    if ultrasonic_cm is not None and 5 < ultrasonic_cm < 400:
        for sname in ['left', 'front', 'right']:
            if sectors[sname] is None:
                # Use ultrasonic reading with sector-specific adjustments
                if sname == 'front':
                    sectors[sname] = {'z': ultrasonic_cm, 'track': None, 'confidence': 0.8}
                else:
                    # Left/right sectors: assume slightly further than front
                    sectors[sname] = {'z': ultrasonic_cm * 1.2, 'track': None, 'confidence': 0.6}
    
    D_MAX = 400.0
    for sname, info in sectors.items():
        if info is None:
            continue
        z = info['z']

    info = {
        "frame": frame,
        "dets": det_list,
        "tracks": tracks,
        "sector_dist": {s: (info['z'] if info else 999) for s, info in sectors.items()},
        "ultrasonic_cm": ultrasonic_cm,
        "vision_cm": np.median(vision_depths) if vision_depths else -1
    }
    return info
