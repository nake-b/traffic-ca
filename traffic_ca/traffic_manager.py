import random
from typing import Iterable, Optional

import numpy as np

from traffic_ca.entity.accident import Accident
from traffic_ca.entity.car import Car
from traffic_ca.utils import random_bool


class TrafficManager:
    def __init__(
        self,
        highway_len: int,
        max_velocity: int,
        car_entry_probability: float,
        slowdown_probability: float,
        init_density: float,
        accident: Optional[Accident] = None,
    ):
        self.highway_len = highway_len
        self.max_velocity = max_velocity
        self.car_entry_probability = car_entry_probability
        self.slowdown_probability = slowdown_probability
        self.init_density = init_density
        self.accident = accident
        self.__car_entry_queue = []
        self.cars = self.__init_random_cars()
        self.__last_t = 0

    def update_traffic(self, t: int) -> None:
        self.__update_accident(t)
        self.__update_cars()
        self.__update_car_entry_queue()

    @property
    def car_positions(self) -> list[int]:
        return [car.position for car in self.cars]

    @property
    def car_velocities(self) -> np.ndarray[int]:
        """
        List with values representing velocity of car in the highway position
        """
        vels = np.zeros(self.highway_len)
        for car in self.cars:
            vels[car.position] = car.velocity
        return vels

    def __update_accident(self, t: int) -> None:
        if self.accident is None:
            return
        self.accident.update(t)

    def __update_cars(self) -> None:
        for car, next_car in self.__iter_consecutive_cars():
            self.__update_car(car, next_car)

    def __iter_consecutive_cars(
        self, cars: Optional[list[Car]] = None
    ) -> Iterable[tuple[Car, Optional[Car]]]:
        if cars is None:
            cars = self.cars
        cars.sort(key=lambda car: car.position)
        next_cars = cars[1:] + [None]
        return zip(cars, next_cars)

    def __update_car(self, car: Car, next_car: Optional[Car]):
        # Acceleration
        car.accelerate()
        # Deceleration
        reaction_objects = [next_car]
        if self.accident is not None and self.accident.occurring:
            reaction_objects.append(self.accident)
        for reaction_object in reaction_objects:
            car.react(reaction_object)
        # Randomization
        if random_bool(self.slowdown_probability):
            car.decelerate()
        # Move
        car.move()
        if car.position >= self.highway_len:
            self.cars.remove(car)

    def __update_car_entry_queue(self) -> None:
        # Randomly add car to queue
        if random_bool(self.car_entry_probability):
            new_car = Car(position=0, velocity=1, max_velocity=self.max_velocity)
            self.__car_entry_queue.append(new_car)
        # If there is room at the beginning, enter a car
        minimal_highway_car_position = min(car.position for car in self.cars)
        if self.__car_entry_queue and minimal_highway_car_position > 0:
            first_car_in_q = self.__car_entry_queue.pop()
            self.cars.append(first_car_in_q)

    def __init_random_cars(self) -> list[Car]:
        # Random initial positions
        all_positions = range(self.highway_len)
        num_cars = int(self.init_density * self.highway_len)
        car_positions = random.sample(all_positions, num_cars)
        # Random initial velocities
        car_velocities = [random.randint(0, self.max_velocity) for _ in range(num_cars)]
        cars = self.__gen_cars(car_positions, car_velocities)
        # Decrease velocities if necessary
        for car, next_car in self.__iter_consecutive_cars(cars):
            car.react(next_car)
        return cars

    def __gen_cars(self, positions: list[int], velocities: list[int]) -> list[Car]:
        return [
            Car(pos, vel, self.max_velocity) for pos, vel in zip(positions, velocities)
        ]
