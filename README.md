# JAXTPC

## Overview

JAXTPC is a high-performance Time Projection Chamber (TPC) simulation framework built with JAX. The system models particle interactions in liquid argon TPCs, including drift, diffusion, recombination, and signal formation on wire planes.

## Repository Structure

- `tools/`: Core simulation modules
  - `drift.py`: Electron drift calculations
  - `geometry.py`: Detector geometry definition
  - `loader.py`: Data loading utilities
  - `recombination.py`: Charge recombination models
  - `responses.py`: Wire response functions
  - `simulation.py`: Main simulation engine
  - `visualization.py`: Visualization tools
  - `wires.py`: Wire signal calculations
- `config/`: Detector configuration files
- `run_simulation.ipynb`: Notebook for executing simulations

## Usage

The simulation can be run using the provided Jupyter notebook `run_simulation.ipynb`.

## Features

- GPU-accelerated TPC simulation using JAX
- Electron drift with diffusion effects
- Angle-dependent signal induction
- Electron lifetime attenuation
- Box model recombination
- Wire plane response convolution
- Visualization tools for signal analysis
