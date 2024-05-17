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

            # compute advantages
            with torch.no_grad():
                next_value = nn_agent.get_value(torch.tensor(next_observations[nn_agent.agent_name]).to(self.agent_config.device).flatten())
                advantages = torch.zeros_like(rollout_data.rewards)
                lastgaelam = 0
                for t in reversed(range(len(rollout_data))):
                    if t == self.agent_config.buffer_size - 1:
                        nextnonterminal = 1.0
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - terminations[nn_agent.agent_name]
                        nextvalues = rollout_data.values[t + 1]
                    delta = rollout_data.rewards[t] + self.agent_config.gamma * nextvalues * nextnonterminal - rollout_data.values[t]
                    advantages[t] = lastgaelam = delta + self.agent_config.gamma * self.agent_config.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + rollout_data.values

                for i in range(len(rollout_data.returns)):
                    rollout_data.returns[i] = returns[i]
                    rollout_data.advantages[i] = advantages[i]

            clipfracs = []
            metrics = {'value_losses': [],
                       'policy_losses': [],
                       'entropy_losses': [],
                       'old_approx_kls': [],
                       'approx_kls': [],
                       'clipfracs': []}

            for _ in range(self.agent_config.update_steps):
                # shuffle rollout_data
                shuffled_inds = np.random.permutation(np.arange(self.agent_config.buffer_size))
                batch_inds_list = [shuffled_inds[i:i+self.agent_config.batch_size] for i in range(0, self.agent_config.buffer_size, self.agent_config.batch_size)]

                batches = [
                    IPPO_Buffer_Samples(
                        rollout_data.observations[batch_inds],
                        rollout_data.actions[batch_inds],
                        rollout_data.rewards[batch_inds],
                        rollout_data.episode_numbers[batch_inds],
                        rollout_data.log_probs[batch_inds],
                        rollout_data.values[batch_inds],
                        rollout_data.advantages[batch_inds],
                        rollout_data.returns[batch_inds]
                    ) for batch_inds in batch_inds_list
                ]

                for data in batches:
                    _, newlogprob, entropy, newvalue = nn_agent.get_action_and_value(data.observations, data.actions)
                    logratio = newlogprob - data.log_probs.squeeze()
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.agent_config.clip_coef).float().mean().item()]

                    advantages = data.advantages.squeeze()

                    if self.agent_config.normalize_advantages:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.agent_config.clip_coef, 1 + self.agent_config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.agent_config.clip_vloss:
                        v_loss_unclipped = (newvalue - data.returns.squeeze()) ** 2
                        v_clipped = data.values.squeeze() + torch.clamp(
                            newvalue - data.values.squeeze(),
                            -self.agent_config.clip_coef,
                            self.agent_config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - data.returns.squeeze()) ** 2
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
                        log_prob
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
