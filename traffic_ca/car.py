from dataclasses import dataclass
from typing import Optional


@dataclass
class Car1D:
    position: int
    speed: int
    max_speed: int

    def move(self, cells: Optional[int] = None) -> None:
        if cells is None:
            cells = self.speed
        self.position += cells

    def distance_to(self, other_car: "Car1D") -> int:
        return abs(self.position - other_car.position)

    def accelerate(self, units: int = 1):
        if self.speed < self.max_speed:
            self.speed += units

    def decelerate(self, units: int = 1):
        units = units if units < self.speed else 0
        self.accelerate(-units)

    def set_speed(self, val: int):
        self.speed = val
