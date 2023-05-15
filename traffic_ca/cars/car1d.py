from dataclasses import dataclass
from typing import Optional


@dataclass
class Car1D:
    position: int
    velocity: int
    max_velocity: int

    def move(self, num_cells: Optional[int] = None) -> None:
        if num_cells is None:
            num_cells = self.velocity
        self.position += num_cells

    def distance_to(self, other_car: "Car1D") -> int:
        return abs(self.position - other_car.position)

    def accelerate(self, units: int = 1):
        new_velocity = min(self.max_velocity, self.velocity + units)
        self.set_velocity(new_velocity)

    def decelerate(self, units: int = 1):
        new_velocity = max(0, self.velocity - units)
        self.set_velocity(new_velocity)

    def set_velocity(self, val: int):
        if val < 0:
            raise ValueError("Velocity must be non-negative.")
        self.velocity = val
