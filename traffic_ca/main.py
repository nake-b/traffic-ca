from traffic_ca.automaton import TrafficCA
from traffic_ca.entity.accident import Accident

if __name__ == "__main__":
    tca = TrafficCA(
        accident=Accident(position=100, start_timestep=100, duration=80),
        timesteps=600,
    )
    tca.animate()
