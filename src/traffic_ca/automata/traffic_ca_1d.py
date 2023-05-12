import cellpylib as cpl
from dataclasses import dataclass
from typing import Callable


@dataclass
class Car1D:
    position: int
    speed: int

    def move(self, cells: int) -> None:
        self.position += cells

    def distance_to(self, other_car: "Car1D") -> int:
        return abs(self.position - other_car.position)


class TrafficCA1D:
    def __init__(self, l: int = 40, w: int = 40):
        self.l = l
        self.w = w
        self.__row = l // 2
        self.__cars = self.__gen_cars([0, 7, 20], [3, 2, 1])
        self.__ca = self.__init_ca
        self.__last_t = 0
        self.__anim = None
        self.__evolve()

    def __evolve(self, timesteps: int = 20):
        self.__ca = cpl.evolve2d(self.__ca, timesteps=timesteps,
                                 apply_rule=self.__rule,
                                 r=self.w, neighbourhood="von Neumann")

    def __rule(self, n, c, t) -> bool:
        self.__update(t)
        y, x = c
        if y != self.__row:
            return 0
        for car in self.__cars:
            if car.position == x:
                return True
        return False

    def __update(self, t) -> None:
        if t == self.__last_t:
            return
        self.__last_t = t
        self.__update_cars()

    def __update_cars(self) -> None:
        self.__cars.sort(key=lambda car: car.position)
        for car, next_car in zip(self.__cars, self.__cars[1:]):
            dist = car.distance_to(next_car)
            if dist < car.speed:
                car.move(dist)
            else:
                car.move(car.speed)
        last_car = self.__cars[-1]
        last_car.move(last_car.speed)

    @property
    def __init_ca(self):
        ca = np.zeros((self.l, self.w))
        with np.nditer(ca, op_flags=['readwrite'], flags=['multi_index']) as it:
            for cell in it:
                y, x = it.multi_index
                if y != self.__row:
                    cell[...] = 0
                    continue
                for car in self.__cars:
                    if car.position == x:
                        cell[...] = 1
        return np.array([ca])

    def __gen_cars(self, positions: list[int], speeds: list[int]) -> list[Car1D]:
        return [Car1D(pos, spd) for pos, spd in zip(positions, speeds)]

    def animate(self, **kwargs):
        self.__anim = cpl.plot2d_animate(self.__ca, **kwargs)
        print(self.__anim.fig)
        return self.__anim

if __name__ == "__main__":
    TrafficCA1D().animate()