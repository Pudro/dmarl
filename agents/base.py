from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Base_Agent(torch.nn.Module):
    def __init__(self, agent_config: Namespace, env: MAgentEnv, agent_name: str, device: Optional[Union[str, int, torch.device]]) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.agent_name = agent_name
        self.env = env
        self.build_network()
        self.device = device

    def build_network(self):
        input_dim = np.array(self.env.observation_spaces[self.agent_name].shape).prod()
        hidden_dims = self.agent_config.hidden_layers
        output_dim = self.env.action_spaces[self.agent_name].n

        layer_sizes = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        layers = [
            nn.Linear(input_dim, hidden_dims[0], device=self.device),
            nn.ReLU()
        ]
        # TODO: make the activation and network_type automagically from config
        #
        for in_size, out_size in layer_sizes:
            layers.append(nn.Linear(in_size, out_size, device=self.device))
            layers.append(nn.ReLU())

        breakpoint()
        layers.append(nn.Linear(hidden_dims[-1], output_dim, device=self.device))
        self.network = nn.Sequential(
            *layers
        )
        breakpoint()
            # nn.Linear(np.array(env.observation_spaces[agent].shape).prod(), 120),
            # nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            # nn.Linear(84, env.action_spaces[agent].n),

        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
