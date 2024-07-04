# About

This repository contains the code and data used for experiments as part of my master thesis on Deep Multi-Agent Reniforcement Learning. The experiments were done utilising the [MAgent2](https://github.com/Farama-Foundation/MAgent2) library.

# Example Usage

```bash
python main.py --config <path/to/config>
```

# Structure

- `agents` contains classes representing individual agents in the environment, and the internal policy representation
- `buffers` contains replay and rollout buffers of agents using a given algorithm
- `configs` contains examplary config files
- `environments` contains custom environments defined by the user
- `runner` contains the runner class which handles data between the trainers and the environment
- `trainers` contains classes which handle interactions of agents in a given group with the environment

# Installation

Python Version: 3.10.9

```bash
pip install -r ./requirements.txt
```
