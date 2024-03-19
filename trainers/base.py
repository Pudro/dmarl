from argparse import Namespace
from agents import AGENT_REGISTRY
from environments.magent_env import MAgentEnv
import torch.nn as nn
import torch


class Base_Trainer:
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.side_name = self.agent_config.side_name
        self.algorithm = self.agent_config.algorithm
        self.env = env
        self.nn_agents = self._make_agents()

    def step(self):
        raise NotImplementedError

    def get_action_futures(self, observations) -> dict[str, torch.Future]:
        # return [
        #     torch.jit.fork(
        #         nn_agent,
        #         torch.Tensor(observations[nn_agent.agent_name]).to(
        #             self.agent_config.device
        #         ),
        #     )
        #     for nn_agent in self.nn_agents
        # ]

        raise NotImplementedError

    def update_agents(
        self,
        global_step,
        actions,
        observations,
        next_observations,
        rewards,
        infos,
        terminations,
        writer,
    ):
        raise NotImplementedError

    def save_agents(self):
        raise NotImplementedError

    def load_agents(self):
        raise NotImplementedError

    def _make_agents(self):
        agent_list = []

        for agent_name in self.env.agents:
            stripped_agent_name = agent_name.split("_")[0]

            if self.side_name == stripped_agent_name:
                agent_list.append(
                    AGENT_REGISTRY[self.algorithm](
                        self.agent_config, self.env, agent_name
                    )
                )

        return nn.ModuleList(agent_list)
