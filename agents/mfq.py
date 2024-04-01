from numpy import who
import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy
import torch.nn as nn
import torch.optim as optim
from buffers import MFQ_Buffer


class MFQ_Agent(Base_Agent):
    def __init__(
        self,
        agent_config: Namespace,
        env: MAgentEnv,
        agent_name: str,
    ) -> None:
        self.agent_name = agent_name
        self.agent_config = agent_config
        self.env = env
        self.device = self.agent_config.device

        super().__init__(agent_config, env, agent_name)

        self.rb = MFQ_Buffer(
            self.agent_config.buffer_size,
            self.env.observation_spaces[self.agent_name],
            self.env.action_spaces[self.agent_name],
            str(self.device),
            handle_timeout_termination=False,
        )

        self.target_network = copy.deepcopy(self.network)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.to(self.device)

        input_dim = self.env.action_spaces[self.agent_name].n
        hidden_dims = self.agent_config.hidden_mean_layers
        output_dim = self.env.action_spaces[self.agent_name].n
        mean_network_layers = self._get_network_layers(input_dim, hidden_dims, output_dim)
        self.mean_network = nn.Sequential(*mean_network_layers)

        self.target_mean_network = copy.deepcopy(self.mean_network)
        self.target_mean_network.load_state_dict(self.mean_network.state_dict())
        self.target_mean_network.to(self.device)

        self.cat_layer = nn.Linear(output_dim*2, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)
        self.to(self.device)

    def forward(self, observations: torch.Tensor, actions_mean: torch.Tensor):
        mean_values = self.mean_network(actions_mean)
        agent_values = self.network(observations)

        return self.cat_layer(torch.cat((agent_values, mean_values), dim=-1))
    
    def target(self, observations: torch.Tensor, actions_mean: torch.Tensor):
        mean_values = self.target_mean_network(actions_mean)
        agent_values = self.target_network(observations)

        return self.cat_layer(torch.cat((agent_values, mean_values), dim=-1))
