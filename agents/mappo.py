import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy


class MAPPO_Agent(Base_Agent):
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

        self.target_network = copy.deepcopy(self.network)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.to(self.device)

        raise NotImplementedError
