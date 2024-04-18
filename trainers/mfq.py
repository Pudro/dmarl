from supersuit.multiagent_wrappers import black_death_v3
from trainers.base import Base_Trainer
from environments.magent_env import MAgentEnv
from argparse import Namespace
import os
import torch
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFQ_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        self.temperature = self.agent_config.start_temperature
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

        self.old_mean_action_probs = torch.zeros(self.env.action_spaces[f'{self.side_name}_0'].n).to(self.agent_config.device)

    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        # TODO: may move these functions outside
        def _full_action_future():
            return torch.argmax(
                    nn_agent(
                        torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                            self.agent_config.device
                        ),
                    self.old_mean_action_probs.to(self.agent_config.device)
                    ),
                    # dim=0,
            )

        action_futures = {}
        for nn_agent in self.nn_agents:
            if random.random() < self.epsilon:
                action_fut = torch.jit.fork(
                    lambda: torch.tensor(
                        self.env.action_space(nn_agent.agent_name).sample()
                    ).to(self.agent_config.device)
                )
            else:
                action_fut = torch.jit.fork(
                    _full_action_future
                )

            action_futures[nn_agent.agent_name] = action_fut

        return {
            agent: torch.jit.wait(fut)
            for agent, fut in action_futures.items()
        }

    def get_boltzmann_policy(self, x):
        return F.softmax((x / self.temperature), dim=-1)

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
        def _update_mfq():
            data = nn_agent.rb.sample(self.agent_config.batch_size)

            with torch.no_grad():
                target_max, indices = nn_agent.target(data.next_observations, data.mean_next_actions).max(dim=1)
                pi = self.get_boltzmann_policy(target_max)
                v_mf = target_max * pi
                td_target = (
                    data.rewards.flatten()
                    + self.agent_config.gamma * v_mf * (1 - data.dones.flatten())
                )

            old_val = nn_agent(data.observations, data.mean_actions).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 1 == 0:
                writer.add_scalar(f"td_loss/{nn_agent.agent_name}", loss, global_step)
                writer.add_scalar(
                    f"q_values/{nn_agent.agent_name}",
                    old_val.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    f"reward/{nn_agent.agent_name}",
                    rewards[nn_agent.agent_name],
                    global_step,
                )
                writer.add_scalar(
                    f"epsilon_greedy/{nn_agent.agent_name}",
                    self.epsilon,
                    global_step,
                )
                writer.add_scalar(
                    f"temperature/{nn_agent.agent_name}",
                    self.temperature,
                    global_step,
                )

            # optimize the model
            nn_agent.optimizer.zero_grad()
            loss.backward()
            nn_agent.optimizer.step()

            # update target network
            if global_step % self.agent_config.target_network_train_period == 0:
                for target_network_param, q_network_param in zip(
                    nn_agent.target_network.parameters(), nn_agent.parameters()
                ):
                    target_network_param.data.copy_(
                        self.agent_config.tau * q_network_param.data
                        + (1.0 - self.agent_config.tau) * target_network_param.data
                    )

        rb_futures = []

        side_actions = {k: v for k, v in actions.items() if self.side_name in k}
        one_hot_next_actions = F.one_hot(torch.stack([*side_actions.values()]), num_classes=self.env.action_spaces[f'{self.side_name}_0'].n).float()
        mean_next_actions = torch.mean(one_hot_next_actions, dim=0)

        for nn_agent in self.nn_agents:
            rb_futures.append(
                torch.jit.fork(
                    nn_agent.rb.add,
                    observations[nn_agent.agent_name],
                    next_observations[nn_agent.agent_name],
                    actions[nn_agent.agent_name].cpu(),
                    np.array(rewards[nn_agent.agent_name]),
                    np.array(terminations[nn_agent.agent_name]),
                    infos[nn_agent.agent_name],
                    self.old_mean_action_probs.cpu(),
                    mean_next_actions.cpu()
                )
            )

        for fut in rb_futures:
            torch.jit.wait(fut)

        update_futures = []
        if global_step > self.agent_config.learning_start:
            if global_step % self.agent_config.train_period == 0:
                for nn_agent in self.nn_agents:
                    update_futures.append(torch.jit.fork(_update_mfq))

        for fut in rb_futures:
            torch.jit.wait(fut)

        self.greedy_decay(global_step)
        self.temperature_decay(global_step)

        self.old_mean_action_probs = mean_next_actions
    

    def save_agents(self, checkpoint=None):

        save_path = self.agent_config.model_dir_save + "/" + self.agent_config.side_name
        if checkpoint:
            save_path = save_path + "/" + str(checkpoint)

        if (not os.path.exists(save_path)) and (not self.agent_config.test_mode):
            os.makedirs(save_path)

        for nn_agent in self.nn_agents:
            torch.save(
                {
                    "agent_config": nn_agent.agent_config,
                    "agent_name": nn_agent.agent_name,
                    "network_state_dict": nn_agent.network.state_dict(),
                    "target_network_state_dict": nn_agent.target_network.state_dict(),
                    "mean_network_state_dict": nn_agent.mean_network.state_dict(),
                    "target_mean_network_state_dict": nn_agent.target_mean_network.state_dict(),
                    "cat_layer_state_dict": nn_agent.cat_layer.state_dict(),
                    "optimizer_state_dict": nn_agent.optimizer.state_dict(),
                },
                save_path + f"/{nn_agent.agent_name}.tar",
            )

    def load_agents(self):
        model_files = sorted(
            os.listdir(self.agent_config.model_dir_load),
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(model_files) < len(self.nn_agents):
            raise Exception(
                f"Model directory {self.agent_config.model_dir_load} has fewer files than needed"
            )

        for nn_agent, file_name in zip(self.nn_agents, model_files):
            model_path = self.agent_config.model_dir_load + f"/{file_name}"
            model_tar = torch.load(model_path, map_location=self.agent_config.device)
            nn_agent.network.load_state_dict(model_tar["network_state_dict"])
            if not self.agent_config.load_policy_only:
                nn_agent.target_network.load_state_dict(model_tar["target_network_state_dict"])
                nn_agent.mean_network.load_state_dict(model_tar["mean_network_state_dict"])
                nn_agent.target_mean_network.load_state_dict(model_tar["target_mean_network_state_dict"])
                nn_agent.cat_layer.load_state_dict(model_tar["cat_layer_state_dict"])

            if not self.agent_config.reset_optimizer:
                nn_agent.optimizer.load_state_dict(model_tar["optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)

    def temperature_decay(self, global_step):
        temperature_step = (self.agent_config.start_temperature - self.agent_config.end_temperature) / self.agent_config.temperature_decay_steps
        self.temperature = max(self.agent_config.end_temperature, self.agent_config.start_temperature - temperature_step * global_step)
