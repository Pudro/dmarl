from agents.ippo import IPPO_Agent
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


class IPPO_Trainer(Base_Trainer):
    def __init__(self, agent_config: Namespace, env: MAgentEnv) -> None:
        self.agent_config = agent_config
        self.env = env
        self.epsilon = self.agent_config.start_greedy
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

    def get_actions(self, observations, infos) -> dict[str, torch.Future]:
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
        def _update_ippo():
            nn_agent.rb.compute_returns_and_advantage(torch.tensor(nn_agent.rb.values[-1]), terminations[nn_agent.agent_name])
            rb_gen = nn_agent.rb.get(self.agent_config.batch_size)

            clipfracs = []
            metrics = {'value_losses': [],
                       'policy_losses': [],
                       'entropy_losses': [],
                       'old_approx_kls': [],
                       'approx_kls': [],
                       'clipfracs': []}

            for data in rb_gen:
                _, newlogprob, entropy, newvalue = nn_agent.get_action_and_value(data.observations, data.actions)
                logratio = newlogprob - data.old_log_prob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.agent_config.clip_coef).float().mean().item()]

                advantages = data.advantages
                if self.agent_config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.agent_config.clip_coef, 1 + self.agent_config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.agent_config.clip_vloss:
                    v_loss_unclipped = (newvalue - data.returns) ** 2
                    v_clipped = data.old_values + torch.clamp(
                        newvalue - data.old_values,
                        -self.agent_config.clip_coef,
                        self.agent_config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - data.returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - data.returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.agent_config.entropy_coef * entropy_loss + v_loss * self.agent_config.vf_coef

                nn_agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(nn_agent.parameters(), self.agent_config.max_grad_norm)
                nn_agent.optimizer.step()

                metrics['value_losses'].append(v_loss.item())
                metrics['policy_losses'].append(pg_loss.item())
                metrics['entropy_losses'].append(entropy_loss.item())
                metrics['old_approx_kls'].append(old_approx_kl.item())
                metrics['approx_kls'].append(approx_kl.item())
                metrics['clipfracs'].append(np.mean(clipfracs))

            nn_agent.rb.reset()
            nn_agent.rb.full = False

            if global_step % 1 == 0:
                writer.add_scalar(f"value_loss/{nn_agent.agent_name}", np.mean(metrics['value_losses']), global_step)
                writer.add_scalar(f"policy_loss/{nn_agent.agent_name}",  np.mean(metrics['policy_losses']), global_step)
                writer.add_scalar(f"entropy_loss/{nn_agent.agent_name}", np.mean(metrics['entropy_losses']), global_step)
                writer.add_scalar(f"old_approx_kl/{nn_agent.agent_name}", np.mean(metrics['old_approx_kls']), global_step)
                writer.add_scalar(f"approx_kl/{nn_agent.agent_name}", np.mean(metrics['approx_kls']), global_step)
                writer.add_scalar(f"clipfrac/{nn_agent.agent_name}", np.mean(metrics['clipfracs']), global_step)
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

        rb_futures = []
        for nn_agent in self.nn_agents:
            with torch.no_grad():
                # Compute the metrics needed for the RolloutBuffer
                _, log_prob, _, value = nn_agent.get_action_and_value(torch.tensor(observations[nn_agent.agent_name].flatten()).to(self.agent_config.device))
                
                rb_futures.append(
                    torch.jit.fork(
                        nn_agent.rb.add,
                        observations[nn_agent.agent_name],
                        actions[nn_agent.agent_name].cpu(),
                        np.array(rewards[nn_agent.agent_name]),
                        0,
                        value,
                        log_prob
                    )
                )

        for fut in rb_futures:
            torch.jit.wait(fut)

        if global_step > self.agent_config.learning_start:
            if global_step % self.agent_config.train_period == 0:
                update_futures = []
                for nn_agent in self.nn_agents:
                    if nn_agent.rb.full:
                        update_futures.append(torch.jit.fork(_update_ippo))

        for fut in rb_futures:
            torch.jit.wait(fut)

        self.greedy_decay(global_step, infos)
    

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
                    "actor_state_dict": nn_agent.actor.state_dict(),
                    "critic_state_dict": nn_agent.critic.state_dict(),
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
            nn_agent.actor.load_state_dict(model_tar["actor_state_dict"])
            nn_agent.critic.load_state_dict(
                model_tar["critic_state_dict"]
            )

            if not self.agent_config.reset_optimizer:
                nn_agent.optimizer.load_state_dict(model_tar["optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)
