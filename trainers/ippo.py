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
        self.last_episode = 0
        self.last_global_step = 0
        super().__init__(agent_config, env)

        if self.agent_config.model_dir_load:
            self.load_agents()

    def get_actions(self, observations, infos) -> dict[str, torch.Future]:
        action_futures = {}
        for nn_agent in self.nn_agents:
            action_fut = torch.jit.fork(
                nn_agent.choose_action,
                torch.tensor(observations[nn_agent.agent_name].flatten()).to(self.agent_config.device))

            action_futures[nn_agent.agent_name] = action_fut

        result_tuples = {agent_name: torch.jit.wait(fut) for agent_name, fut in action_futures.items()}

        actions = {
            agent_name: torch.tensor(tup[0]).to(self.agent_config.device) for agent_name,
            tup in result_tuples.items()
        }
        # ugly hack to get actions, probs and values all through runner back to trainer
        actions.setdefault(self.side_name, {}).update(result_tuples)

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
            clipfracs = []
            for step in range(self.agent_config.update_steps):
                (obs_arr,
                 action_arr,
                 old_prob_arr,
                 vals_arr,
                 reward_arr,
                 dones_arr,
                 batches) = nn_agent.rb.generate_batches()

                values = vals_arr

                advantage = self.calculate_advantages(values,
                                                      reward_arr,
                                                      dones_arr,
                                                      self.agent_config.gamma,
                                                      self.agent_config.gae_lambda)

                if self.agent_config.normalize_advantages:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                advantage = torch.tensor(advantage).to(self.agent_config.device)
                values = torch.tensor(values).to(self.agent_config.device)
                for batch in batches:
                    obs = torch.tensor(obs_arr[batch], dtype=torch.float).to(self.agent_config.device)
                    old_probs = torch.tensor(old_prob_arr[batch]).to(self.agent_config.device)
                    actions = torch.tensor(action_arr[batch]).to(self.agent_config.device)

                    dist = nn_agent.actor(obs)

                    entropy = dist.entropy()
                    entropy_loss = entropy.mean()
                    critic_value = nn_agent.critic(obs)
                    critic_value = torch.squeeze(critic_value)
                    new_probs = dist.log_prob(actions)
                    # prob_ratio = new_probs.exp() / old_probs.exp()
                    logratio = new_probs - old_probs
                    prob_ratio = (logratio).exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((prob_ratio - 1) - logratio).mean()
                        clipfracs += [((prob_ratio - 1.0).abs() > self.agent_config.clip_coef).float().mean().item()]

                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio,
                                                         1 - self.agent_config.clip_coef,
                                                         1 + self.agent_config.clip_coef) * advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]

                    if self.agent_config.clip_vloss:
                        critic_loss_unclipped = (returns - critic_value)**2
                        critic_clipped = values[batch] + torch.clamp(
                            critic_value - values[batch],
                            -self.agent_config.clip_coef,
                            self.agent_config.clip_coef,
                        )
                        critic_loss_clipped = (critic_clipped - returns)**2
                        critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                        critic_loss = 0.5 * critic_loss_max.mean()
                    else:
                        critic_loss = (returns - critic_value)**2
                        critic_loss = critic_loss.mean() * 0.5

                    total_loss = actor_loss + critic_loss * self.agent_config.vf_coef - self.agent_config.entropy_coef * entropy_loss

                    nn_agent.actor_optimizer.zero_grad()
                    nn_agent.critic_optimizer.zero_grad()
                    total_loss.backward()
                    nn_agent.actor_optimizer.step()
                    nn_agent.critic_optimizer.step()

                if self.agent_config.target_kl is not None and approx_kl > self.agent_config.target_kl:
                    print(f'Target kl: {self.agent_config.target_kl} exceeded at step: {step}. Stopping updates')
                    break

            y_pred, y_true = values[batch].cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            nn_agent.rb.clear_memory()
            if hasattr(self.agent_config, 'entropy_coef_decay_type'):
                self.decay_entropy_coef(global_step, infos)
            print('entropy_coef:', self.agent_config.entropy_coef)

            if global_step % 1 == 0:
                writer.add_scalar(f"total_loss/{nn_agent.agent_name}", total_loss, global_step)
                writer.add_scalar(f"value_loss/{nn_agent.agent_name}", critic_loss, global_step)
                writer.add_scalar(f"policy_loss/{nn_agent.agent_name}", actor_loss, global_step)
                writer.add_scalar(f"entropy_loss/{nn_agent.agent_name}", entropy_loss, global_step)
                writer.add_scalar(f"old_approx_kl/{nn_agent.agent_name}", old_approx_kl, global_step)
                writer.add_scalar(f"approx_kl/{nn_agent.agent_name}", approx_kl, global_step)
                writer.add_scalar(f"explained_variance/{nn_agent.agent_name}", explained_var, global_step)
                writer.add_scalar(
                    f"reward/{nn_agent.agent_name}",
                    rewards[nn_agent.agent_name],
                    global_step,
                )

        rb_futures = []
        for nn_agent in self.nn_agents:
            prob = actions[self.side_name][nn_agent.agent_name][1]
            val = actions[self.side_name][nn_agent.agent_name][2]
            rb_futures.append(
                torch.jit.fork(nn_agent.rb.store_memory,
                               observations[nn_agent.agent_name],
                               actions[nn_agent.agent_name].cpu(),
                               prob,
                               val,
                               np.array(rewards[nn_agent.agent_name]),
                               terminations[nn_agent.agent_name]))

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
                    "actor_state_dict": nn_agent.actor_net.state_dict(),
                    "critic_state_dict": nn_agent.critic_net.state_dict(),
                    "actor_optimizer_state_dict": nn_agent.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": nn_agent.critic_optimizer.state_dict(),
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
            raise Exception(f"Model directory {self.agent_config.model_dir_load} has fewer files than needed")

        for nn_agent, file_name in zip(self.nn_agents, model_files):
            model_path = self.agent_config.model_dir_load + f"/{file_name}"
            model_tar = torch.load(model_path, map_location=self.agent_config.device)
            nn_agent.actor_net.load_state_dict(model_tar["actor_state_dict"], strict=False)
            # this was changed from critic_state_dict
            critic_dict = nn_agent.critic_net.state_dict()
            critic_dict.update({
                k: v for k,
                v in model_tar["critic_state_dict"].items() if k in critic_dict and v.size() == critic_dict[k].size()
            })
            nn_agent.critic_net.load_state_dict(critic_dict, strict=False)

            if not self.agent_config.reset_optimizer:
                nn_agent.critic_optimizer.load_state_dict(model_tar["critic_optimizer_state_dict"])
                nn_agent.actor_optimizer.load_state_dict(model_tar["actor_optimizer_state_dict"])

            nn_agent.to(self.agent_config.device)

    def remember(self, obs, action, probs, vals, reward, done):
        self.rb.store_memory(obs, action, probs, vals, reward, done)

    def calculate_advantages(self, values, rewards, dones, gamma, gae_lambda):

        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards) - 1)):
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        return advantages

    def decay_entropy_coef(self, global_step, infos):
        decay_function = self._get_decay_function(self.agent_config.entropy_coef_decay_type)
        self.agent_config.entropy_coef = decay_function(global_step,
                                                        infos,
                                                        self.agent_config.entropy_coef_start,
                                                        self.agent_config.entropy_coef_end,
                                                        self.agent_config.entropy_coef_steps,
                                                        self.agent_config.entropy_coef_rate)


def test_adv_integrity():

    values = np.array([
        0.72891333,
        1.72100516,
        1.74748425,
        0.58914075,
        2.21380207,
        0.92961213,
        0.45700146,
        -0.8943487,
        -0.71089311,
        1.13792162
    ])
    rewards = np.array([
        0.74076531,
        0.15569448,
        1.13257646,
        0.08365263,
        -0.32540634,
        -1.64755961,
        -0.06796294,
        0.44458574,
        -0.96501062,
        0.57117792
    ])
    dones = np.full(10, False)
    gamma = 0.99
    gae_lambda = 0.95

    expected_advantages = np.array(
        [0.38654917,
         -1.4131823,
         -1.6777045,
         -1.7501818,
         -3.6537561,
         -2.1635978,
         -0.04131586,
         1.4556658,
         0.8724249,
         0.])
    advantages = IPPO_Trainer.calculate_advantages(None, values, rewards, dones, gamma, gae_lambda)

    np.testing.assert_allclose(advantages, expected_advantages, rtol=1e-3)
