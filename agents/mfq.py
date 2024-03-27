import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy
import torch.nn as nn


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

        original_in_features = self.network[0].in_features
        original_out_features = self.network[0].out_features
        n_team_agents = sum([1 for a in self.env.agents if a.split('_')[0] == self.agent_config.side_name])
        new_in_features = original_in_features + n_team_agents
        self.network[0] = nn.Linear(new_in_features, original_out_features, device=self.device)

        self.target_network = copy.deepcopy(self.network)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.to(self.device)

    def forward(self, observations: torch.Tensor, actions_mean: torch.Tensor):
        q_inputs = torch.concat([observations, actions_mean], dim=-1)
        breakpoint()
        return self.network(q_inputs)
