import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from stable_baselines3.common.buffers import RolloutBuffer
from buffers import IPPO_Buffer


class IPPO_Agent(Base_Agent):
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

        self.actor = copy.deepcopy(self.network)
        self.actor.load_state_dict(self.network.state_dict())

        input_dim = np.array(self.env.observation_spaces[self.agent_name].shape).prod()
        hidden_dims = self.agent_config.hidden_layers
        output_dim = 1 # value estimation
        critic_layers = self._get_network_layers(input_dim, hidden_dims, output_dim)
        self.critic = nn.Sequential(*critic_layers)

        del self.network

        # self.rb = RolloutBuffer(
        #     self.agent_config.buffer_size,
        #     self.env.observation_spaces[self.agent_name],
        #     self.env.action_spaces[self.agent_name],
        #     str(self.device),
        #     self.agent_config.gae_lambda,
        #     self.agent_config.gamma
        # )

        self.rb = IPPO_Buffer(
            self.agent_config.buffer_size,
            self.env.observation_spaces[self.agent_name],
            self.env.action_spaces[self.agent_name],
            str(self.device),
            self.agent_config.gae_lambda,
            self.agent_config.gamma
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)
        self.to(self.device)

    def get_action_and_value(self, observations, action=None):
        logits = self.actor(observations)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(observations)

    def get_action(self, observations):
        logits = self.actor(observations)
        probs = Categorical(logits=logits)
        return probs.sample()
