from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import importlib
from stable_baselines3.common.buffers import ReplayBuffer


class Base_Agent(torch.nn.Module):
    def __init__(
        self,
        agent_config: Namespace,
        env: MAgentEnv,
        agent_name: str,
    ) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.agent_name = agent_name
        self.env = env
        self.build_network()
        self.device = self.agent_config.device

    def build_network(self):
        # TODO: make this work for RNNs
        # currently only Linear layers allowed
        
        input_dim = np.array(self.env.observation_spaces[self.agent_name].shape).prod()
        hidden_dims = self.agent_config.hidden_layers
        output_dim = self.env.action_spaces[self.agent_name].n

        layers = self._get_network_layers(input_dim, hidden_dims, output_dim)
        self.network = nn.Sequential(*layers)

        # TODO: consider getting optimizer from config
        self.optimizer = optim.Adam(self.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)

        self.rb = ReplayBuffer(
            self.agent_config.buffer_size,
            self.env.observation_spaces[self.agent_name],
            self.env.action_spaces[self.agent_name],
            str(self.device),
            handle_timeout_termination=False,
        )

        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _get_network_layers(self, input_dim, hidden_dims, output_dim) -> list[nn.Module]:
        Layer_Type = self._get_layer_type()
        Activation = self._get_activation_type()

        layer_sizes = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        layers = [
            Layer_Type(input_dim, hidden_dims[0], device=self.device),
            Activation(),
        ]

        for in_size, out_size in layer_sizes:
            layers.append(Layer_Type(in_size, out_size, device=self.device))
            layers.append(Activation())

        layers.append(Layer_Type(hidden_dims[-1], output_dim, device=self.device))

        return layers

    def _get_layer_type(self):

        NNModule = importlib.import_module("torch.nn")

        return getattr(NNModule, self.agent_config.layer_type)

    def _get_activation_type(self):

        NNModule = importlib.import_module("torch.nn")

        return getattr(NNModule, self.agent_config.activation)
