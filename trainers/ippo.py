from torch.distributions.categorical import Categorical
from agents.ippo import IPPO_Agent
from buffers.ippo import IPPO_Buffer_Samples
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
        self.last_episode = 0
        self.last_global_step = 0
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

    def get_actions(self, observations, infos) -> dict[str, torch.Future]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            action_fut = torch.jit.fork(
                nn_agent.get_action,
                torch.tensor(observations[nn_agent.agent_name].flatten()).to(
                    self.agent_config.device
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
            rollout_data = nn_agent.rb.sample()

            with torch.no_grad():
                next_values = nn_agent.critic(torch.tensor(observations[nn_agent.agent_name]).to(self.agent_config.device))
                advantages = torch.zeros_like(rollout_data.rewards)
                last_gae_lam = 0
                for t in reversed(range(len(advantages))):
                    if t == self.agent_config.buffer_size - 1:
                        nextvalues = next_values
                    else:
                        nextvalues = rollout_data.values[t + 1]
                    delta = rollout_data.rewards[t] + (1 - rollout_data.terminations[t]) * self.agent_config.gamma * nextvalues - rollout_data.values[t]
                    advantages[t] = last_gae_lam = delta + (1 - rollout_data.terminations[t]) * self.agent_config.gamma * self.agent_config.gae_lambda * last_gae_lam
                returns = advantages + rollout_data.values

                for i in range(len(rollout_data.returns)):
                    rollout_data.returns[i] = returns[i]
                    rollout_data.advantages[i] = advantages[i]

            batches = [rollout_data[i:i+self.agent_config.batch_size] for i in range(0, len(rollout_data.values), self.agent_config.batch_size)]

            batches = [
                IPPO_Buffer_Samples(
                    rollout_data.observations[i:i+self.agent_config.batch_size],
                    rollout_data.actions[i:i+self.agent_config.batch_size],
                    rollout_data.rewards[i:i+self.agent_config.batch_size],
                    rollout_data.episode_numbers[i:i+self.agent_config.batch_size],
                    rollout_data.log_probs[i:i+self.agent_config.batch_size],
                    rollout_data.values[i:i+self.agent_config.batch_size],
                    rollout_data.advantages[i:i+self.agent_config.batch_size],
                    rollout_data.returns[i:i+self.agent_config.batch_size],
                    rollout_data.terminations[i:i+self.agent_config.batch_size]
                ) for i in range(0, len(rollout_data.values), self.agent_config.batch_size)
            ]

            clipfracs = []
            metrics = {'value_losses': [],
                       'policy_losses': [],
                       'entropy_losses': [],
                       'old_approx_kls': [],
                       'approx_kls': [],
                       'clipfracs': []}

            # === For data in batches ...
            
            for data in batches:
                # actor loss
                pi_dist = Categorical(logits=nn_agent.actor(data.observations))
                log_pi = pi_dist.log_prob(data.actions)
                ratio = torch.exp(log_pi - data.log_probs)
                surrogate1 = ratio * data.advantages
                surrogate2 = torch.clip(ratio, 1 - self.agent_config.clip_coef, 1 + self.agent_config.clip_coef) * data.advantages
                loss_a = -torch.sum(torch.min(surrogate1, surrogate2), dim=-2, keepdim=True).mean()

                # entropy loss
                entropy = pi_dist.entropy()
                loss_e = entropy.mean()

                # critic loss
                value_pred = nn_agent.actor(data.observations)
                value_target = data.returns
                if self.agent_config.clip_vloss:
                    value_clipped = data.values + (value_pred - data.values).clamp(-self.agent_config.clip_coef, self.agent_config.clip_coef)
                    loss_v = (value_pred - value_target) ** 2
                    loss_v_clipped = (value_clipped - value_target) ** 2
                    loss_c = torch.max(loss_v, loss_v_clipped)
                    loss_c = loss_c.sum()
                else:
                    loss_v = ((value_pred - value_target) ** 2)
                    loss_c = loss_v.sum()


                loss = loss_a + self.agent_config.vf_coef * loss_c - self.agent_config.entropy_coef * loss_e
                nn_agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_agent.parameters(), self.agent_config.max_grad_norm)
                nn_agent.optimizer.step()

                metrics['value_losses'].append(loss_c.item())
                metrics['policy_losses'].append(loss_a.item())
                metrics['entropy_losses'].append(loss_e.item())
                # metrics['old_approx_kls'].append(old_approx_kl.item())
                # metrics['approx_kls'].append(approx_kl.item())
                metrics['clipfracs'].append(np.mean(clipfracs))

            nn_agent.rb.reset()
            nn_agent.rb.full = False

            y_pred, y_true = rollout_data.values.cpu().numpy(), rollout_data.returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if global_step % 1 == 0:
                writer.add_scalar(f"value_loss/{nn_agent.agent_name}", np.mean(metrics['value_losses']), global_step)
                writer.add_scalar(f"policy_loss/{nn_agent.agent_name}",  np.mean(metrics['policy_losses']), global_step)
                writer.add_scalar(f"entropy_loss/{nn_agent.agent_name}", np.mean(metrics['entropy_losses']), global_step)
                writer.add_scalar(f"old_approx_kl/{nn_agent.agent_name}", np.mean(metrics['old_approx_kls']), global_step)
                writer.add_scalar(f"approx_kl/{nn_agent.agent_name}", np.mean(metrics['approx_kls']), global_step)
                writer.add_scalar(f"clipfrac/{nn_agent.agent_name}", np.mean(metrics['clipfracs']), global_step)
                writer.add_scalar(f"explained_variance/{nn_agent.agent_name}", explained_var, global_step)
                writer.add_scalar(
                    f"reward/{nn_agent.agent_name}",
                    rewards[nn_agent.agent_name],
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
                        infos['episode'],
                        value,
                        log_prob,
                        np.array(terminations[nn_agent.agent_name]).astype(float)
                    )
                )

        for fut in rb_futures:
            torch.jit.wait(fut)

        if global_step % self.agent_config.buffer_size == 0:
            update_futures = []
            for nn_agent in self.nn_agents:
                update_futures.append(torch.jit.fork(_update_ippo))

        for fut in rb_futures:
            torch.jit.wait(fut)


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
        all_files = os.listdir(self.agent_config.model_dir_load)
        model_files = [f for f in all_files if os.path.isfile(os.path.join(self.agent_config.model_dir_load, f))]
        model_files = sorted(
            model_files,
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
