from typing import Optional

import cellpylib as cpl
import matplotlib.pyplot as plt
import numpy as np

from traffic_ca.entity.accident import Accident
from traffic_ca.traffic_manager import TrafficManager
from traffic_ca.utils import plot1d_animate

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
        normalized_attrs = {
            "car_entry_probability": car_entry_probability,
            "slowdown_probability": slowdown_probability,
            "init_density": init_density,
        }
        for name, attr in normalized_attrs.items():
            self.__check_normalized_float_attr(name, attr)
        self.__traffic_manager = TrafficManager(
            highway_len=highway_len,
            max_velocity=max_velocity,
            car_entry_probability=car_entry_probability,
            slowdown_probability=slowdown_probability,
            init_density=init_density,
            accident=accident,
        )
        self.__ca = self.__init_ca()
        self.__velocity_matrix = self.__init_velocity_matrix()
        self.__anim = None
        # Finally
        self.__last_timestep = 0
        self.__evolve()

    def __evolve(self) -> None:
        """
        Evolves the internal cellpylib cellular automaton
        """
        self.__ca = cpl.evolve(
            self.__ca,
            timesteps=self.timesteps,
            apply_rule=self.__cpl_update_rule,
            r=self.highway_len,
        )

    def __cpl_update_rule(self, _, cell_idx: int, timestep: int) -> bool:
        """
        Value for cpl.evolve apply_rule parameter
        :param _:  cell's neighbourhood - NOT USED IN STCA
        :param cell_idx: index of the cell in the CA
        :param timestep: current timestep, from [0, self.timestep]
        :return: state of the cell wit the index "cell_idx" in the timestep "timestep"
        """
        if timestep != self.__last_timestep:
            self.__last_timestep = timestep
            self.__traffic_manager.update_traffic(timestep)
            self.__update_velocity_matrix()
        return cell_idx in self.__traffic_manager.car_positions

    def __init_ca(self):
        ca = np.zeros(self.highway_len)
        for idx in self.__traffic_manager.car_positions:
            ca[idx] = 1
        return np.array([ca])

    def __init_velocity_matrix(self) -> list[np.ndarray]:
        return [self.__traffic_manager.car_velocities]

    def __update_velocity_matrix(self):
        vels = self.__traffic_manager.car_velocities
        self.__velocity_matrix.append(vels)

    def animate(self, **kwargs):
        self.__anim = plot1d_animate(self.__ca, **kwargs)
        return self.__anim

    def plot_space_time(self):
        self.__plot_over_time(self.__ca.T, "Active cells", cmap=plt.get_cmap("viridis"))

    def plot_velocity_time(self):
        matrix = np.array(self.__velocity_matrix).T
        scale_coef = 100 / self.max_velocity
        matrix *= scale_coef
        self.__plot_over_time(matrix, "Velocity", cmap=plt.get_cmap("inferno"))

    def __plot_over_time(self, vals: np.array, ylabel: str, **kwargs):
        plt.style.use("classic")
        plt.imshow(vals, **kwargs)
        plt.xlabel("Timestep")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel.capitalize()} over time plot", y=1.05)
        plt.show()

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
        self,
        Y: list | np.ndarray,
        label: str,
        ylim: Optional[tuple[float, float]] = None,
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
        if self.accident is not None and self.accident.start_timestep < self.timesteps:
            end_timestep = min(self.accident.end_timestep, self.timesteps)
            plt.axvspan(
                self.accident.start_timestep,
                end_timestep,
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

    def __check_normalized_float_attr(self, name: str, val: float) -> None:
        if not (0 <= val <= 1):
            raise ValueError(
                f"{name.capitalize()} must be between 0 and 1, please provide a valid value."
            )
