import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
import os
import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from stable_baselines3.common.buffers import RolloutBuffer
from buffers import MAPPO_Buffer


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

        self.actor_net = copy.deepcopy(self.network)
        self.actor_net.load_state_dict(self.network.state_dict())

        del self.network

        self.rb = MAPPO_Buffer(env, agent_config)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)

        self.to(self.device)

    def actor(self, obs):
        dist = self.actor_net(obs)
        dist = Categorical(F.softmax(dist, dim=-1))

        return dist

    def critic(self, state):
        value = self.critic_net(state)

        return value

    def choose_action(self, observation, state):
        dist = self.actor(observation)
        cat_state = torch.cat([observation, state])
        value = self.critic(cat_state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def register_value_network(self, network, optimizer):
        self.critic_net = network
        self.critic_optimizer = optimizer
