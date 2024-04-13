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


class ISAC_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

    def get_actions(self, observations, infos) -> dict[str, torch.Tensor]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            if random.random() < self.epsilon:
                action_fut = torch.jit.fork(
                    lambda: torch.tensor(
                        self.env.action_space(nn_agent.agent_name).sample()
                    )
                )
            else:
                action_fut = torch.jit.fork(
                    torch.argmax,
                    nn_agent.get_action(
                        torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                            self.agent_config.device
                        ),
                    ),
                    # dim=0,
                )

            action_futures[nn_agent.agent_name] = action_fut

        actions = {
            agent_name: torch.jit.wait(fut)
            for agent_name, fut in action_futures.items()
        }

        return actions


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
        def _update_isac():
            data = nn_agent.rb.sample(self.agent_config.batch_size)


            # CRITIC TRAINING
            with torch.no_grad():
                next_state_log_pi, next_state_action_probs = nn_agent.get_action_probs(data.next_observations) # TODO: give actor method?
                qf1_next_target = nn_agent.target_qf1(data.next_observations)
                qf2_next_target = nn_agent.target_qf2(data.next_observations)
                # we can use the action probabilities instead of MC sampling to estimate the expectation
                min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - nn_agent.agent_config.alpha * next_state_log_pi
                )
                # adapt Q-target for discrete Q-function
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * nn_agent.agent_config.gamma * (min_qf_next_target)

             # use Q-values only for the taken actions
            qf1_values = nn_agent.qf1(data.observations)
            qf2_values = nn_agent.qf2(data.observations)
            qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            nn_agent.q_optimizer.zero_grad()
            qf_loss.backward()
            nn_agent.q_optimizer.step()

            # ACTOR TRAINING
            log_pi, action_probs = nn_agent.get_action_probs(data.observations)
            with torch.no_grad():
                qf1_values = nn_agent.qf1(data.observations)
                qf2_values = nn_agent.qf2(data.observations)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_loss = (action_probs * ((nn_agent.agent_config.alpha * log_pi) - min_qf_values)).mean()

            nn_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn_agent.actor_optimizer.step()

            # TODO: consider addin this in
            # if args.autotune:
            #     # re-use action probabilities for temperature loss
            #     alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
            #     a_optimizer.zero_grad()
            #     alpha_loss.backward()
            #     a_optimizer.step()
            #     alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % self.agent_config.target_network_train_period == 0:
                for param, target_param in zip(nn_agent.qf1.parameters(), nn_agent.target_qf1.parameters()):
                    target_param.data.copy_(self.agent_config.tau * param.data + (1 - self.agent_config.tau) * target_param.data)
                for param, target_param in zip(nn_agent.qf2.parameters(), nn_agent.target_qf2.parameters()):
                    target_param.data.copy_(self.agent_config.tau * param.data + (1 - self.agent_config.tau) * target_param.data)

            if global_step % 1 == 0:
                writer.add_scalar(f"qf1_values/{nn_agent.agent_name}", qf1_a_values.mean().item(), global_step)
                writer.add_scalar(f"qf2_values/{nn_agent.agent_name}", qf2_a_values.mean().item(), global_step)
                writer.add_scalar(f"qf1_loss/{nn_agent.agent_name}", qf1_loss.item(), global_step)
                writer.add_scalar(f"qf2_loss/{nn_agent.agent_name}", qf2_loss.item(), global_step)
                writer.add_scalar(f"qf_loss/{nn_agent.agent_name}", qf_loss.item() / 2.0, global_step)
                writer.add_scalar(f"actor_loss/{nn_agent.agent_name}", actor_loss.item(), global_step)
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


        # TODO: currently data is added to the replay buffer on each global_step
        # maybe add the whole episode to replay buffer each episode
        # and add episodic returns to each reward
        rb_futures = []
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
                )
            )

        for fut in rb_futures:
            torch.jit.wait(fut)

        update_futures = []
        if global_step > self.agent_config.learning_start:
            if global_step % self.agent_config.train_period == 0:
                for nn_agent in self.nn_agents:
                    update_futures.append(torch.jit.fork(_update_isac))

        for fut in rb_futures:
            torch.jit.wait(fut)

        self.greedy_decay(global_step)
    

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
                    "qf1_state_dict": nn_agent.qf1.state_dict(),
                    "target_qf1_state_dict": nn_agent.target_qf1.state_dict(),
                    "qf2_state_dict": nn_agent.qf2.state_dict(),
                    "target_qf2_state_dict": nn_agent.target_qf2.state_dict(),
                    "actor_state_dict": nn_agent.actor.state_dict(),
                    "q_optimizer_state_dict": nn_agent.q_optimizer.state_dict(),
                    "actor_optimizer_state_dict": nn_agent.actor_optimizer.state_dict(),
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
            nn_agent.qf1.load_state_dict(model_tar["qf1_state_dict"])
            nn_agent.target_qf1.load_state_dict(model_tar["target_qf1_state_dict"])
            nn_agent.qf2.load_state_dict(model_tar["qf2_state_dict"])
            nn_agent.target_qf2.load_state_dict(model_tar["target_qf2_state_dict"])
            nn_agent.actor.load_state_dict(model_tar["actor_state_dict"])

            if not self.agent_config.reset_optimizer:
                nn_agent.q_optimizer.load_state_dict(model_tar["q_optimizer_state_dict"])
                nn_agent.actor_optimizer.load_state_dict(model_tar["actor_optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)
