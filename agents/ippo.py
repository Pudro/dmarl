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
from buffers import IPPO_Buffer


class PPOMemory:

    def __init__(self, batch_size):
        self.obs = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.obs)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.obs),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.obs.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.obs = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


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

        self.actor_net = copy.deepcopy(self.network)
        self.actor_net.load_state_dict(self.network.state_dict())

        input_dim = np.array(self.env.observation_spaces[self.agent_name].shape).prod()
        hidden_dims = self.agent_config.hidden_layers
        output_dim = 1    # value estimation
        critic_layers = self._get_network_layers(input_dim, hidden_dims, output_dim)
        self.critic_net = nn.Sequential(*critic_layers)

        del self.network

        self.rb = PPOMemory(self.agent_config.batch_size)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.agent_config.learning_rate, eps=1e-5)

        self.to(self.device)

    def actor(self, obs):
        dist = self.actor_net(obs)
        dist = Categorical(F.softmax(dist, dim=-1))

        return dist

    def critic(self, obs):
        value = self.critic_net(obs)

        return value

    def choose_action(self, observation):
        dist = self.actor(observation)
        value = self.critic(observation)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
