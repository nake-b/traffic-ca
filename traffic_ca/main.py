from traffic_ca.automaton import TrafficCA
from traffic_ca.entity.accident import Accident

if __name__ == "__main__":
    TrafficCA(
        highway_len=500,
        init_density=0.4,
        slowdown_probability=0.3,
        car_entry_probability=0.7,
        accident=Accident(position=150, start_timestep=100, duration=100),
        max_velocity=5,
        timesteps=600,
    ).animate()
