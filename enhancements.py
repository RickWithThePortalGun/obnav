from __future__ import annotations
from typing import Dict, Tuple

def suggest_path(info: Dict, dist_close_cm: float = 80.0) -> Dict[str, object]:
    sd = info.get("sector_dist", {"left": 999.0, "front": 999.0, "right": 999.0}).copy()
    u  = float(info.get("ultra_cm", sd.get("front", 999.0)) or 999.0)
    if u < 999:
        sd["front"] = min(sd.get("front", 999.0), u)
    safest = max(sd, key=sd.get)
    front_d = sd.get("front", 999.0)

    if front_d >= dist_close_cm:
        path = "forward"
    else:
        path = "left" if sd.get("left", 0) >= sd.get("right", 0) else "right"

    nearest_sector = min(sd, key=sd.get)
    nearest_d = sd[nearest_sector]
    if nearest_d >= 200:
        buzz_ms = 60
    elif nearest_d >= 120:
        buzz_ms = 90
    elif nearest_d >= 60:
        buzz_ms = 130
    else:
        buzz_ms = 200

    return {
        "path": path,
        "target_sector": nearest_sector,
        "buzz_ms": int(buzz_ms),
        "sector_dist": sd,
    }
