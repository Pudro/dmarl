from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import torch.optim as optim
import os
import copy
from buffers.qmix import QMIX_Buffer
import numpy as np


class QMIX_Agent(Base_Agent):

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

        self.rb = QMIX_Buffer(
            self.agent_config.buffer_size,
            self.env.observation_spaces[self.agent_name],
            self.env.action_spaces[self.agent_name],
            np.prod(self.env.state_space.shape),
            str(self.device),
            handle_timeout_termination=False,
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)
        self.to(self.device)
