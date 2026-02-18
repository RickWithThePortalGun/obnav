# Obnav Split-Compute Demo

## Components
- `server.py`: ZMQ REP server running YOLOv8. Receives JPEG bytes, returns boxes `[x1,y1,x2,y2,conf,cls]`.
- `adapters_sim.py`: Sim adapters (camera, detector, rangefinder, haptics) and drawing utils.
- `adapters_prod_remote.py`: Remote detector adapter that calls the server.
- `core.py`: Fusion and decision logic. Exposes `sector_from_box` and `step_once`.
- `run_sim.py`: All-in-one simulation with local YOLO.
- `run_prod_remote.py`: Pi-side loop. Uses remote detector.
- `client_smoke.py`: One-shot client that sends one frame and prints boxes.
- `ports.py`: Protocols for Camera, Detector, Rangefinder, Haptics.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run: local simulation
```bash
python run_sim.py
```

## Run: split compute
Terminal A (host with GPU/CPU):
```bash
YOLO_WEIGHTS=yolov8n.pt SERVER_PORT=5555 python server.py
```
Terminal B (edge/Pi), set HOST_IP:
```bash
export HOST_IP=192.168.1.50
python run_prod_remote.py
```

## Smoke test
```bash
python client_smoke.py
```

## Environment
- `CONF_THRESH` default 0.35
- `DIST_CLOSE_CM` default 80
- `OBSTACLE_CLASSES` default `0,56,60` (COCO person, chair, table)
- `LEFT_SPLIT` default 0.30
- `RIGHT_SPLIT` default 0.70
- `SERVER_PORT` default 5555
- `YOLO_WEIGHTS` default `yolov8n.pt`
- `CONF` default 0.35
- `IMG` default 416
- `OBSTACLE_NAMES` default `person,chair,bench,table`

## Notes
- ZeroMQ is unauthenticated. Use on trusted LAN.
- Camera index in `VideoCamera(src=0)` may need change.
- Press `Esc` in server window to exit.
# obnav
