# About

This repository contains the code and data used for experiments as part of my master thesis.

# Installation

There may be a problem with installing MAgent2 using `pip`. Issue can be found at https://github.com/Farama-Foundation/MAgent2/issues/19

If so:
- create and activate a local environment
- clone MAgent2: `git clone https://github.com/Farama-Foundation/MAgent2`
- run `pip install ./MAgent2`

I had to downgrade pettingzoo to 1.22.3 for python 3.11.5 (`env.reset()` not working properly)
