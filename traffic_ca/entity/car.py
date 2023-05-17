from dataclasses import dataclass
from typing import Optional

from traffic_ca.entity.road_object import RoadObject


@dataclass
class Car:
    position: int
    velocity: int
    max_velocity: int

    def move(self, num_cells: Optional[int] = None) -> None:
        if num_cells is None:
            num_cells = self.velocity
        self.position += num_cells

    def distance_to(self, other: RoadObject) -> int:
        return abs(self.position - other.position)

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

    def react(self, other: Optional[RoadObject]) -> None:
        if other is None:  # No reaction
            return
        # Else max velocity is the number of free cells in front
        dist = self.distance_to(other)
        new_velocity = min(dist - 1, self.velocity)
        new_velocity = max(0, new_velocity)
        self.set_velocity(new_velocity)
