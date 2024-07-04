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

There may be a problem with installing MAgent2 using `pip`. Issue can be found at https://github.com/Farama-Foundation/MAgent2/issues/19

If so:
- create and activate a local environment
- clone MAgent2: `git clone https://github.com/Farama-Foundation/MAgent2`
- run `pip install ./MAgent2`

I had to downgrade pettingzoo to 1.22.3 for python 3.11.5 (`env.reset()` not working properly)
