from trainers.base import Base_Trainer
from environments.magent_env import MAgentEnv
from argparse import Namespace
import torch
import random
import numpy as np


class IQL_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        super().__init__(agent_config, env)

    def get_action_futures(self, observations) -> dict[str, torch.Future]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            if random.random() < self.epsilon:
                action_fut = torch.jit.fork(
                    lambda: torch.tensor(self.env.action_space(nn_agent.agent_name).sample())
                )
            else:
                # q_values = nn_agent(torch.Tensor(observations[nn_agent.agent_name].flatten()).to(self.agent_config.device))
                # action = torch.argmax(q_values, dim=0)
                action_fut = torch.jit.fork(
                    torch.argmax,
                    nn_agent(
                        torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                            self.agent_config.device
                        ),
                    ),
                    # dim=0,
                )

            action_futures[nn_agent.agent_name] = action_fut

        # return {
        #     nn_agent.agent_name: torch.jit.fork(
        #         nn_agent,
        #         torch.Tensor(observations[nn_agent.agent_name]).to(
        #             self.agent_config.device
        #         ),
        #     )
        #     for nn_agent in self.nn_agents
        # }

        return action_futures

    def update_agents(self):
        # self.epsilon = ...
        raise NotImplementedError
