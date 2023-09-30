# traffic-ca
Modelling traffic using cellular automata. Currently only supports NaSch (STCA) model with car entry queue and soft boundary.
## Background
Open source library written as the final project "Modelling traffic using Cellular Automata", for the course "Mathematical Modelling and Simulation", at Prirodno-matematički fakultet, Univerzitet u Tuzli.\
Uses [cellpylib](https://github.com/lantunes/cellpylib) -  *Antunes, L. M. (2021). CellPyLib: A Python Library for working with Cellular Automata. Journal of Open Source Software, 6(67), 3608.*

## How to install
1. Create and activate conda environment.
```shell
conda create -n traffic_ca python=3.10
conda activate traffic_ca
```
2. Install the package.
```shell
pip install -e .
```

## How to use
Sample usage can be found in `traffic_ca/main.py`. Run `python traffic_ca/main.py` for an example STCA animation.
## Citation
The repo can be cited as:
> Nadir Bašić (2023). TrafficCA: Modelling Traffic using Cellular Automata. Prirodno-matematički fakultet, Univerzitet u Tuzli

