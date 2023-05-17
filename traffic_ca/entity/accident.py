from dataclasses import dataclass


@dataclass
class Accident:
    position: int
    start_timestep: int
    duration: int
    occurring: bool = False

    def __post_init__(self):
        self.end_timestep = self.start_timestep + self.duration

    def update(self, timestep: int):
        if self.occurring and timestep > self.end_timestep:
            self.occurring = False
            return
        if not self.occurring and self.start_timestep <= timestep <= self.end_timestep:
            self.occurring = True
