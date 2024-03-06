from cv2 import goodFeaturesToTrack
from environments import MAgentEnv
from torch.utils.tensorboard.writer import SummaryWriter
from agents import AGENT_REGISTRY
from gymnasium.spaces.box import Box
from pathlib import Path
import numpy as np
import socket
import wandb
import time
import os

class Runner:
    def __init__(self, configs) -> None:

        self.configs = configs if type(configs) == list else configs
        self.fps = self.configs[0].fps
        time_string = time.asctime().replace(" ", "").replace(":", "_")

        # TODO:
        # this assumes that there might be multiple config files (one for each agents)
        # maybe it is better to create one config file for both agents
        # (leverage yaml syntax)
        for config in self.configs:
            seed = f"seed_{config.seed}_"
            config.model_dir_load = config.model_dir
            config.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)

            if (not os.path.exists(config.model_dir_save)) and (not config.test_mode):
                os.makedirs(config.model_dir_save)

            if config.logger == "tensorboard":
                log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                self.writer = SummaryWriter(log_dir)
                self.use_wandb = False
            else:
                self.use_wandb = True

        for config in self.configs:
            if config.agent_name == "random":
                continue
            else:
                self.configs_base = config

                # NOTE: here goes the init
                # Runner Base init
                self.env = self._make_env(config)
                self.env.reset()

                self.running_steps = config.running_steps
                self.training_frequency = config.training_frequency
                self.train_per_step = config.train_per_step

                # build environments
                self.n_handles = len(self.env.handles)
                self.agent_keys = self.env.agents
                self.agent_ids = self.env.agent_ids
                # self.agent_keys_all = self.env.keys
                self.n_agents_all = self.env.n_agents_all
                self.render = config.render

                self.n_steps = config.running_steps
                self.test_mode = config.test_mode
                self.marl_agents, self.marl_names = [], []
                self.current_step, self.current_episode = 0, 0

                if self.use_wandb:
                    config_dict = vars(config)
                    wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
                    if not wandb_dir.exists():
                        os.makedirs(str(wandb_dir))
                    wandb.init(config=config_dict,
                               project=config.project_name,
                               entity=config.wandb_user_name,
                               notes=socket.gethostname(),
                               dir=wandb_dir,
                               group=config.env_id,
                               job_type=config.agent,
                               name=time.asctime(),
                               reinit=True)
                break

        # self.episode_length = self.env.max_episode_length

        # environment details, representations, policies, optimizers, and agents.
        for h, config in enumerate(self.configs):
            config.handle_name = self.env.side_names[h]
            if self.n_handles > 1 and config.agent != "RANDOM":
                config.model_dir += "{}/".format(config.handle_name)
            config.handle, config.n_agents = h, self.env.n_agents[h]
            config.agent_keys, config.agent_ids = self.agent_keys[h], self.agent_ids[h]
            config.state_space = self.env.state_space
            config.observation_space = self.env.observation_space
            if isinstance(self.env.action_space[self.agent_keys[h][0]], Box):
                config.dim_act = self.env.action_space[self.agent_keys[h][0]].shape[0]
                config.act_shape = (config.dim_act,)
            else:
                config.dim_act = self.env.action_space[self.agent_keys[h][0]].n
                config.act_shape = ()
            config.action_space = self.env.action_space
            if config.env_name == "MAgent2":
                config.obs_shape = (np.prod(self.env.observation_space[self.agent_keys[h][0]].shape),)
                config.dim_obs = config.obs_shape[0]
            else:
                config.obs_shape = self.env.observation_space[self.agent_keys[h][0]].shape
                config.dim_obs = config.obs_shape[0]
            config.rew_shape, config.done_shape, config.act_prob_shape = (config.n_agents, 1), (config.n_agents,), (config.dim_act,)
            self.marl_agents.append(AGENT_REGISTRY[config.agent](config, self.env, config.device))
            self.marl_names.append(config.agent)
            if config.test_mode:
                self.marl_agents[h].load_model(config.model_dir, config.seed)

        self.print_infos(self.configs)

    def _make_env(self, config):
        return MAgentEnv(config.env_id, config.seed,
                                    minimap_mode=config.minimap_mode,
                                    max_cycles=config.max_cycles,
                                    extra_features=config.extra_features,
                                    map_size=config.map_size,
                                    render_mode=config.render_mode)

    def print_infos(self, config):
            infos = []
            for h, arg in enumerate(config):
                agent_name = self.agent_keys[h][0][0:-2]
                if arg.n_agents == 1:
                    infos.append(agent_name + ": {} agent".format(arg.n_agents) + ", {}".format(arg.agent))
                else:
                    infos.append(agent_name + ": {} agents".format(arg.n_agents) + ", {}".format(arg.agent))
            print(infos)
            time.sleep(0.01)

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
        else:
            for k, v in info.items():
                self.writer.add_video(k, v, fps=fps, global_step=x_index)

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                wandb.log({k: v}, step=x_index)
        else:
            for k, v in info.items():
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def combine_env_actions(self, actions):
        actions_envs = []
        num_env = actions[0].shape[0]
        for e in range(num_env):
            act_handle = {}
            for h, keys in enumerate(self.agent_keys):
                act_handle.update({agent_name: actions[h][e][i] for i, agent_name in enumerate(keys)})
            actions_envs.append(act_handle)
        return actions_envs

    def get_actions(self, obs_n, test_mode, act_mean_last, agent_mask, state):
        actions_n, log_pi_n, values_n, actions_n_onehot = [], [], [], []
        act_mean_current = act_mean_last
        for h, mas_group in enumerate(self.marl_agents):
            if self.marl_names[h] == "MFQ":
                _, a, a_mean = mas_group.act(obs_n[h], test_mode=test_mode, act_mean=act_mean_last[h], agent_mask=agent_mask[h])
                act_mean_current[h] = a_mean
            elif self.marl_names[h] == "MFAC":
                a, a_mean = mas_group.act(obs_n[h], test_mode, act_mean_last[h], agent_mask[h])
                act_mean_current[h] = a_mean
                _, values = mas_group.values(obs_n[h], act_mean_current[h])
                values_n.append(values)
            elif self.marl_names[h] == "VDAC":
                _, a, values = mas_group.act(obs_n[h], state=state, test_mode=test_mode)
                values_n.append(values)
            elif self.marl_names[h] in ["MAPPO", "IPPO"]:
                _, a, log_pi = mas_group.act(obs_n[h], test_mode=test_mode, state=state)
                _, values = mas_group.values(obs_n[h], state=state)
                log_pi_n.append(log_pi)
                values_n.append(values)
            elif self.marl_names[h] in ["COMA"]:
                _, a, a_onehot = mas_group.act(obs_n[h], test_mode)
                _, values = mas_group.values(obs_n[h], state=state, actions_n=a, actions_onehot=a_onehot)
                actions_n_onehot.append(a_onehot)
                values_n.append(values)
            else:
                _, a = mas_group.act(obs_n[h], test_mode=test_mode)
            actions_n.append(a)
        return {'actions_n': actions_n, 'log_pi': log_pi_n, 'act_mean': act_mean_current,
                'act_n_onehot': actions_n_onehot, 'values': values_n}

    def store_data(self, obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, done_n):
        for h, mas_group in enumerate(self.marl_agents):
            if mas_group.args.agent_name == "random":
                continue
            data_step = {'obs': obs_n[h], 'obs_next': next_obs_n[h], 'actions': actions_dict['actions_n'][h],
                         'state': state, 'state_next': next_state, 'rewards': rew_n[h],
                         'agent_mask': agent_mask[h], 'terminals': done_n[h]}
            if mas_group.on_policy:
                data_step['values'] = actions_dict['values'][h]
                if self.marl_names[h] == "MAPPO":
                    data_step['log_pi_old'] = actions_dict['log_pi'][h]
                elif self.marl_names[h] == "COMA":
                    data_step['actions_onehot'] = actions_dict['act_n_onehot'][h]
                else:
                    pass
                mas_group.memory.store(data_step)
                if mas_group.memory.full:
                    if self.marl_names[h] == "COMA":
                        _, values_next = mas_group.values(next_obs_n[h],
                                                          state=next_state,
                                                          actions_n=actions_dict['actions_n'][h],
                                                          actions_onehot=actions_dict['act_n_onehot'][h])
                    elif self.marl_names[h] == "MFAC":
                        _, values_next = mas_group.values(next_obs_n[h], actions_dict['act_mean'][h])
                    elif self.marl_names[h] == "VDAC":
                        _, _, values_next = mas_group.act(next_obs_n[h])
                    else:
                        _, values_next = mas_group.values(next_obs_n[h], state=next_state)
                    for i_env in range(1):
                        if done_n[h][i_env].all():
                            mas_group.memory.finish_path(0.0, i_env)
                        else:
                            mas_group.memory.finish_path(values_next[i_env], i_env)
                continue
            elif self.marl_names[h] in ["MFQ", "MFAC"]:
                data_step['act_mean'] = actions_dict['act_mean'][h]
            else:
                pass
            mas_group.memory.store(data_step)

    def train_episode(self, n_episodes):
        act_mean_last = [np.zeros([self.n_envs, arg.dim_act]) for arg in self.args]
        terminal_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool_)
        truncate_handle = np.zeros([self.n_handles, self.n_envs], dtype=np.bool_)
        episode_score = np.zeros([self.n_handles, self.n_envs, 1], dtype=np.float32)
        episode_info, train_info = {}, {}
        for _ in tqdm(range(n_episodes)):
            obs_n = self.envs.buf_obs
            state, agent_mask = self.envs.global_state(), self.envs.agent_mask()
            for step in range(self.episode_length):
                actions_dict = self.get_actions(obs_n, False, act_mean_last, agent_mask, state)
                actions_execute = self.combine_env_actions(actions_dict['actions_n'])
                next_obs_n, rew_n, terminated_n, truncated_n, infos = self.envs.step(actions_execute)
                next_state, agent_mask = self.envs.global_state(), self.envs.agent_mask()

                self.store_data(obs_n, next_obs_n, actions_dict, state, next_state, agent_mask, rew_n, terminated_n)

                # train the model for each step
                if self.train_per_step:
                    if self.current_step % self.training_frequency == 0:
                        for h, mas_group in enumerate(self.marl_agents):
                            if mas_group.args.agent_name == "random":
                                continue
                            train_info = self.marl_agents[h].train(self.current_step)

                obs_n, state, act_mean_last = deepcopy(next_obs_n), deepcopy(next_state), deepcopy(
                    actions_dict['act_mean'])

                for h, mas_group in enumerate(self.marl_agents):
                    episode_score[h] += np.mean(rew_n[h] * agent_mask[h][:, :, np.newaxis], axis=1)
                    terminal_handle[h] = terminated_n[h].all(axis=-1)
                    truncate_handle[h] = truncated_n[h].all(axis=-1)

                for i_env in range(self.n_envs):
                    if terminal_handle.all(axis=0)[i_env] or truncate_handle.all(axis=0)[i_env]:
                        self.current_episode[i_env] += 1
                        for h, mas_group in enumerate(self.marl_agents):
                            if mas_group.args.agent_name == "random":
                                continue
                            if mas_group.on_policy:
                                if mas_group.args.agent == "COMA":
                                    _, value_next_e = mas_group.values(next_obs_n[h],
                                                                       state=next_state,
                                                                       actions_n=actions_dict['actions_n'][h],
                                                                       actions_onehot=actions_dict['act_n_onehot'][h])
                                elif mas_group.args.agent == "MFAC":
                                    _, value_next_e = mas_group.values(next_obs_n[h], act_mean_last[h])
                                elif mas_group.args.agent == "VDAC":
                                    _, _, value_next_e = mas_group.act(next_obs_n[h])
                                else:
                                    _, value_next_e = mas_group.values(next_obs_n[h], state=next_state)
                                mas_group.memory.finish_path(value_next_e[i_env], i_env)
                            obs_n[h][i_env] = infos[i_env]["reset_obs"][h]
                            agent_mask[h][i_env] = infos[i_env]["reset_agent_mask"][h]
                            act_mean_last[h][i_env] = np.zeros([self.args[h].dim_act])
                            episode_score[h, i_env] = np.mean(infos[i_env]["individual_episode_rewards"][h])
                        state[i_env] = infos[i_env]["reset_state"]
                self.current_step += self.n_envs

            if self.n_handles > 1:
                for h in range(self.n_handles):
                    episode_info["Train_Episode_Score/side_{}".format(self.args[h].handle_name)] = episode_score[h].mean()
            else:
                episode_info["Train_Episode_Score"] = episode_score[0].mean()

            # train the model for each episode
            if not self.train_per_step:
                for h, mas_group in enumerate(self.marl_agents):
                    if mas_group.args.agent_name == "random":
                        continue
                    train_info = self.marl_agents[h].train(self.current_step)
            self.log_infos(train_info, self.current_step)
            self.log_infos(episode_info, self.current_step)

    def finish(self):
        self.env.close()
        if self.use_wandb:
            wandb.finish()
        else:
            self.writer.close()

    def run(self):
        raise NotImplementedError

