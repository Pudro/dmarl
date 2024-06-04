import torch
from argparse import Namespace
from typing import Optional, Union
from environments.magent_env import MAgentEnv
from agents.base import Base_Agent
from torch.distributions.categorical import Categorical
import os
import copy
import torch.optim as optim
import torch.nn.functional as F
from buffers.qmix import QMIX_Buffer
import numpy as np


class MASAC_Agent(Base_Agent):

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

        self.qf1 = copy.deepcopy(self.network)
        self.qf1.load_state_dict(self.network.state_dict())
        self.qf2 = copy.deepcopy(self.network)
        self.qf2.load_state_dict(self.network.state_dict())

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.actor = copy.deepcopy(self.network)
        self.actor.load_state_dict(self.network.state_dict())

        del self.network
        del self.optimizer

        self.rb = QMIX_Buffer(
            self.agent_config.buffer_size,
            self.env.observation_spaces[self.agent_name],
            self.env.action_spaces[self.agent_name],
            np.prod(self.env.state_space.shape),
            str(self.device),
            handle_timeout_termination=False,
        )

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()),
                                      lr=self.agent_config.learning_rate,
                                      eps=1e-5)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.agent_config.learning_rate, eps=1e-5)

        self.to(self.device)

    def get_action_probs(self, observations):
        logits = self.actor(observations)
        policy_dist = Categorical(logits=logits)
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob, action_probs

    def get_action(self, observations):
        logits = self.actor(observations)
        policy_dist = Categorical(logits=logits)
        return policy_dist.sample()
