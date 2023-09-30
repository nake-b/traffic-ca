import random
from typing import Optional

import cellpylib as cpl
import matplotlib.pyplot as plt
import numpy as np

from traffic_ca.entity.accident import Accident
from traffic_ca.entity.car import Car
from traffic_ca.utils import plot1d_animate, random_bool

CITY_TRAFFIC_SLOWDOWN_PROBABILITY = 0.5
HIGHWAY_TRAFFIC_SLOWDOWN_PROBABILITY = 0.3


class TrafficCA:
    def __init__(
        self,
        highway_len: int = 350,
        max_velocity: int = 5,
        init_density: float = 0.4,
        slowdown_probability: float = HIGHWAY_TRAFFIC_SLOWDOWN_PROBABILITY,
        car_entry_probability: float = 0.7,
        accident: Optional[Accident] = None,
        timesteps: Optional[int] = None,
    ):
        self.highway_len = highway_len
        self.max_velocity = max_velocity
        self.accident = accident
        if timesteps is None:
            timesteps = highway_len
        self.timesteps = timesteps
        self.car_entry_probability = self.__check_normalized_float_attr(
            "car_entry_probability", car_entry_probability
        )
        self.__car_entry_queue = []
        self.slowdown_probability = self.__check_normalized_float_attr(
            "slowdown_probability", slowdown_probability
        )
        self.init_density = self.__check_normalized_float_attr(
            "init_density", init_density
        )
        self.__cars = self.__init_random_cars()
        self.__ca = self.__init_ca()
        self.__velocity_matrix = [np.zeros(self.highway_len)]
        self.__last_t = 0
        self.__anim = None
        # Finally
        self.__evolve()
        self.__velocity_matrix[0] = self.__velocity_matrix[1]

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
        self.__update_accident(t)
        self.__update_cars()
        self.__update_velocity_matrix()

    def __update_car(self, car: Car, next_car: Optional[Car]):
        # Acceleration
        car.accelerate()
        # Deceleration
        reaction_objects = [next_car]
        if self.accident is not None and self.accident.occurring:
            reaction_objects.append(self.accident)
        for _object in reaction_objects:
            car.react(_object)
        # Randomization
        if random_bool(self.slowdown_probability):
            car.decelerate()
        # Move
        car.move()
        if car.position >= self.highway_len:
            self.__cars.remove(car)

    def __update_cars(self) -> None:
        self.__cars.sort(key=lambda car: car.position)
        next_cars = self.__cars[1:] + [None]
        for car, next_car in zip(self.__cars, next_cars):
            self.__update_car(car, next_car)
        self.__update_car_entry_queue()

    def __update_car_entry_queue(self):
        if random_bool(self.car_entry_probability):
            new_car = Car(position=0, velocity=1, max_velocity=self.max_velocity)
            self.__car_entry_queue.append(new_car)
        if self.__car_entry_queue and min(car.position for car in self.__cars) != 0:
            first_car_in_q = self.__car_entry_queue.pop()
            self.__cars.append(first_car_in_q)

    def __update_accident(self, t: int):
        if self.accident is None:
            return
        self.accident.update(t)

    def __init_ca(self):
        ca = np.zeros(self.highway_len)
        for car in self.__cars:
            ca[car.position] = 1
        return np.array([ca])

    def __update_velocity_matrix(self):
        vels = np.zeros(self.highway_len)
        for car in self.__cars:
            vels[car.position] = car.velocity
        self.__velocity_matrix.append(vels)

    def __gen_cars(self, positions: list[int], velocities: list[int]) -> list[Car]:
        return [
            Car(pos, vel, self.max_velocity) for pos, vel in zip(positions, velocities)
        ]

    def __init_random_cars(self) -> list[Car]:
        all_positions = range(self.highway_len)
        num_cars = int(self.init_density * self.highway_len)
        car_positions = random.sample(all_positions, num_cars)
        car_velocities = [random.randint(0, self.max_velocity) for _ in range(num_cars)]
        cars = self.__gen_cars(car_positions, car_velocities)
        return cars

    def animate(self, **kwargs):
        self.__anim = plot1d_animate(self.__ca, **kwargs)
        return self.__anim

    def __plot_over_time(self, vals: np.array, ylabel: str, **kwargs):
        plt.style.use("classic")
        # plt.figure(figsize=(10, 10))
        plt.imshow(vals, **kwargs)
        plt.xlabel("Timestep")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel.capitalize()} over time plot", y=1.05)
        plt.show()

    def plot_space_time(self):
        self.__plot_over_time(self.__ca.T, "Active cells", cmap=plt.get_cmap("viridis"))

    def plot_velocity_time(self):
        matrix = np.array(self.__velocity_matrix).T
        scale_coef = 100 / self.max_velocity
        matrix *= scale_coef
        self.__plot_over_time(matrix, "Velocity", cmap=plt.get_cmap("inferno"))

    def plot_mean_velocity_time(self):
        mean_velocities = self.__compute_mean_velocities()
        self.__plot_scalar_over_time(
            mean_velocities, "Mean velocity", (0, max(mean_velocities) + 0.5)
        )

    def plot_density_time(self):
        self.__plot_scalar_over_time(self.__compute_densities(), "Density", (0, 1))

    def plot_flow_time(self):
        mean_velocities = self.__compute_mean_velocities()
        densities = self.__compute_densities()
        flow_vector = np.array(mean_velocities) * np.array(densities)
        self.__plot_scalar_over_time(flow_vector, "Flow")

    def __compute_densities(self) -> list[float]:
        return [sum(ca_state) / len(ca_state) for ca_state in self.__ca]

    def __compute_mean_velocities(self) -> list[float]:
        return [
            velocity_vector.sum() / sum(ca_state)
            for velocity_vector, ca_state in zip(self.__velocity_matrix, self.__ca)
        ]

    def __plot_scalar_over_time(
        self, Y: list, label: str, ylim: Optional[tuple[float, float]] = None
    ):
        plt.style.use("seaborn-darkgrid")
        X = range(self.timesteps)
        plt.plot(X, Y)
        plt.xlabel("Timestep")
        plt.ylabel(label)
        plt.title(f"{label.capitalize()} over time plot")
        if ylim is not None:
            start, end = ylim
            plt.ylim(start, end)
        if self.accident is not None:
            plt.axvspan(
                self.accident.start_timestep,
                self.accident.end_timestep,
                color="red",
                alpha=0.4,
                label="Accident",
            )
        plt.legend()
        plt.show()

    def plot_space_velocity_time(self):
        plt.style.use("seaborn-darkgrid")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        for t in range(self.timesteps):
            X_positions = self.__ca[t]
            X = np.argwhere(X_positions)
            Y = [t]
            Z_positions = self.__velocity_matrix[t]
            Z = [Z_positions[x] for x in X]
            ax.scatter(X, Y, Z, s=0.6, c="black", marker="s")
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")
        ax.set_zlabel("Velocity")
        plt.show()

    def __check_normalized_float_attr(self, name: str, val: float) -> float:
        if not (0 <= val <= 1):
            raise ValueError(
                f"{name.capitalize()} must be between 0 and 1, please provide a valid value."
            )
        return val
