from __future__ import annotations

from typing import List, Tuple, Protocol
import numpy as np


Box = Tuple[int,int,int,int,float,int]

class Camera(Protocol):
    def read(self) -> np.ndarray: ...

class Detector(Protocol):
    def infer(self, frame: np.ndarray) -> List[Box]: ...

class Rangefinder(Protocol):
    def distance_cm(self) -> float | None: ...

class Haptics(Protocol):
    def buzz(self, channel: str, ms: int = 100) -> None: ...