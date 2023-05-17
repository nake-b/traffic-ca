from dataclasses import dataclass
from typing import Protocol


@dataclass
class RoadObject(Protocol):
    position: int
