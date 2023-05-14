from traffic_ca.automata.traffic_ca_1d import TrafficCA1D

if __name__ == "__main__":
    TrafficCA1D(init_density=0.2, slowdown_probability=0.3).animate(interval=300)
