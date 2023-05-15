import random
from typing import Optional

import cellpylib as cpl
import numpy as np

from traffic_ca.cars.car1d import Car1D
from traffic_ca.utils import plot1d_animate, random_bool

CITY_TRAFFIC_SLOWDOWN_PROBABILITY = 0.5
HIGHWAY_TRAFFIC_SLOWDOWN_PROBABILITY = 0.3


class TrafficCA1D:
    def __init__(
        self,
        highway_len: int = 200,
        max_velocity: int = 5,
        init_density: float = 0.6,
        slowdown_probability: float = HIGHWAY_TRAFFIC_SLOWDOWN_PROBABILITY,
        timesteps: Optional[int] = None,
    ):
        self.highway_len = highway_len
        self.max_velocity = max_velocity
        if timesteps is None:
            timesteps = highway_len
        self.timesteps = timesteps
        if not (0 <= slowdown_probability <= 1):
            raise ValueError(
                f"Slowdown probability must be between 0 and 1, please provide a valid value."
            )
        self.slowdown_probability = slowdown_probability
        if not (0 <= init_density <= 1):
            raise ValueError(
                f"Initial density must be between 0 and 1, please provide a valid value."
            )
        self.init_density = init_density
        self.__cars = self.__gen_random_cars()
        self.__ca = self.__init_ca
        self.__last_t = 0
        self.__anim = None
        self.__evolve()

    def __evolve(self):
        self.__ca = cpl.evolve(
            self.__ca,
            timesteps=self.timesteps,
            apply_rule=self.__rule,
            r=self.highway_len,
        )

    def __rule(self, n, c, t) -> bool:
        self.__update(t)
        for car in self.__cars:
            if car.position == c:
                return True
        return False

    def __update(self, t) -> None:
        if t == self.__last_t:
            return
        self.__last_t = t
        self.__update_cars()

    def __update_car(self, car: Car1D, next_car: Optional[Car1D]):
        # Acceleration
        car.accelerate()
        # Deceleration
        if next_car is None:
            new_speed = car.velocity
        else:
            dist = car.distance_to(next_car)
            new_speed = min(dist, car.velocity)
        car.set_velocity(new_speed)
        # Randomization
        if random_bool(self.slowdown_probability):
            car.decelerate()
        # Move
        car.move()

    def __update_cars(self) -> None:
        self.__cars.sort(key=lambda car: car.position)
        next_cars = self.__cars[1:] + [None]
        for car, next_car in zip(self.__cars, next_cars):
            self.__update_car(car, next_car)

    @property
    def __init_ca(self):
        ca = np.zeros(self.highway_len)
        for car in self.__cars:
            ca[car.position] = 1
        return np.array([ca])

    def __gen_cars(self, positions: list[int], speeds: list[int]) -> list[Car1D]:
        return [
            Car1D(pos, spd, self.max_velocity) for pos, spd in zip(positions, speeds)
        ]

    def __gen_random_cars(self) -> list[Car1D]:
        all_positions = range(self.highway_len)
        num_cars = int(self.init_density * self.highway_len)
        car_positions = random.sample(all_positions, num_cars)
        car_speeds = [random.randint(0, self.max_velocity) for _ in range(num_cars)]
        cars = self.__gen_cars(car_positions, car_speeds)
        return cars

    def animate(self, **kwargs):
        self.__anim = plot1d_animate(self.__ca, **kwargs)
        return self.__anim


if __name__ == "__main__":
    TrafficCA1D().animate()
