import numpy as np
import torch
# from torch import random
from torch.distributions import one_hot_categorical
import time
from common.common import find_neighbor_pos, find_neighbor_id
import random
import pickle


class RolloutWorkerMagent:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.view_field = args.view_field
        # self.pos_action_shape = (args.id_dim + args.n_actions) * args.nei_n_agents
        self.args = args
        # self.idact_shape = args.id_dim + args.n_actions

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode_ma(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents + self.args.more_enemy)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            # if fixed_num_agents < self.n_agents:
            #     self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, maven_z, evaluate)
                else:
                    # st = time.time()
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    # print(time.time()-st)
                if self.args.use_fixed_model:
                    fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    if isinstance(fixed_action, np.int64):
                        fixed_action = fixed_action.astype(np.int32)
                    else:
                        fixed_action = fixed_action.cpu()
                        fixed_action = fixed_action.numpy().astype(np.int32)
                    fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                # acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
                acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy())
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, fixed_rewards

    def generate_episode_ma_test(self, episode_num=None, evaluate=False, var_num=None):
        n_agents = var_num
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=n_agents * 2 * self.args.more_walls)
        self.env.add_agents(handles[0], method="random", n=n_agents)
        self.env.add_agents(handles[1], method="random", n=n_agents + self.args.more_enemy)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((n_agents, self.args.n_actions))
        self.agents.policy.init_hidden_test(1, var_nums=n_agents)
        if self.args.use_fixed_model:
            self.agents.fixed_policy.init_hidden_test(1, var_nums=n_agents)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < n_agents:
                self.env.add_agents(handles[0], method="random", n=n_agents - num_agents)
            # if fixed_num_agents < n_agents:
            #     self.env.add_agents(handles[1], method="random", n=n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

            for j in range(n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            for agent_id in range(n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_action = np.ones(self.n_actions)
                if self.args.alg == 'maven':
                    # action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                    #                                    avail_action, epsilon, maven_z, evaluate)
                    action = self.agents.choose_action_test(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, maven_z, evaluate, var_num=n_agents)
                else:
                    # st = time.time()
                    # action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                    #                                    avail_action, epsilon, evaluate)
                    action = self.agents.choose_action_test(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate, var_num=n_agents)
                    # print(time.time()-st)
                if self.args.use_fixed_model:
                    fixed_action = self.agents.choose_fixed_action(fixed_obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    if isinstance(fixed_action, np.int64):
                        fixed_action = fixed_action.astype(np.int32)
                    else:
                        fixed_action = fixed_action.cpu()
                        fixed_action = fixed_action.numpy().astype(np.int32)
                    fixed_actions.append(fixed_action)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                # acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
                acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
        return [], episode_reward, win_tag, fixed_rewards

    def generate_episode_ma_obs_wvdn(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded, neighbor_list, neighbor_pos_list = [], [], [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents + self.args.more_enemy)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((self.n_agents, self.args.n_actions))
        empty_team = {}
        for i in range(self.n_agents):
            empty_team[i] = []

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            num_agents = self.env.get_num(handles[0])
            # fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)

            obs_all = self.env.get_observation(handles[0])
            # fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))

            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            maven_z_list = [[] for _ in range(self.n_agents)]
            pos = self.env.get_pos(handles[0])
            neighbor_dic, neighbor_pos = find_neighbor_pos(pos, self.args.view_field, self.args.num_neighbor)

            for agent_id in range(self.n_agents):
                avail_action = np.ones(self.n_actions)
                if 'ours' in self.args.alg:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, maven_z_list[agent_id], evaluate)
                else:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                # acts[1] = np.array(np.random.randint(0, self.n_actions, size=self.n_agents, dtype='int32'))
                acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            neighbor_list.append(neighbor_dic)
            neighbor_pos_list.append(neighbor_pos)
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            neighbor_list.append(empty_team)
            neighbor_pos_list.append(empty_team)

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy())
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if 'ours' in self.args.alg:
            episode['neighbor'] = neighbor_list.copy()
            episode['neighbor_pos'] = neighbor_pos_list.copy()
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, fixed_rewards

    def generate_episode_ma_obs_wvdn_test(self, episode_num=None, evaluate=False, var_num=None):
        n_agents = var_num
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        # o, u, r, s, avail_u, u_onehot, terminate, padded, z_list, plot_z_list = [], [], [], [], [], [], [], [], [], []
        # z_team, z_king = [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=n_agents * 2 * self.args.more_walls)
        self.env.add_agents(handles[0], method="random", n=n_agents)
        self.env.add_agents(handles[1], method="random", n=n_agents + self.args.more_enemy)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        fixed_rewards = 0
        last_action = np.zeros((n_agents, self.args.n_actions))
        # if not 'ours' in self.args.alg:
        #     self.agents.policy.init_hidden_test(1, var_nums=n_agents)
        # if self.args.use_fixed_model:
        #     self.agents.fixed_policy.init_hidden_test(1, var_nums=n_agents)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < n_agents:
                self.env.add_agents(handles[0], method="random", n=n_agents - num_agents)
            # if fixed_num_agents < n_agents:
            #     self.env.add_agents(handles[1], method="random", n=n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

            for j in range(n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                # fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            actions, avail_actions, actions_onehot, fixed_actions = [], [], [], []
            neighbor_clean_actions = {}
            need_search_neighbor = []
            maven_z_list = [[] for _ in range(n_agents)]
            pos = self.env.get_pos(handles[0])
            neighbor_dic, neighbor_pos = find_neighbor_pos(pos, self.args.view_field, self.args.num_neighbor)

            for agent_id in range(n_agents):
                avail_action = np.ones(self.n_actions)
                if 'ours' in self.args.alg:
                    if not self.args.use_coll_policy:
                        # print(obs[agent_id], neighbor_dic[agent_id])
                        action = self.agents.choose_action_test(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, maven_z_list[agent_id], evaluate, var_num=var_num)
                    else:
                        others_avail_action = {}
                        if neighbor_dic[agent_id]:
                            for others_id in neighbor_dic[agent_id]:
                                others_avail_action[others_id] = np.ones(self.n_actions)
                        if self.args.use_pilike_coll:
                            action = self.agents.choose_action_ground_true_pilike_coll_dq(obs[agent_id], last_action[agent_id], neighbor_dic, agent_id, avail_action, epsilon, maven_z_list[agent_id],
                                                                                          evaluate, others_avail_action)
                        else:
                            action = self.agents.choose_action_ground_true(obs[agent_id], last_action[agent_id], neighbor_dic, agent_id, avail_action, epsilon, maven_z_list[agent_id], evaluate,
                                                                           others_avail_action)
                    # print(action, epsilon)
                else:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            if self.args.use_fixed_model:
                acts[1] = np.array(fixed_actions)
            else:
                # acts[1] = np.array(np.random.randint(0, self.n_actions, size=n_agents, dtype='int32'))
                acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.
            # z_king.append(neighbor_king)
            episode_reward += reward
            fixed_rewards += fixed_reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return [], episode_reward, win_tag, fixed_rewards


class CommRolloutWorkerMagent:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode_ma(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        handles = self.env.get_handles()
        self.env.add_walls(method="random", n=self.n_agents * 2 * self.args.more_walls)
        self.env.add_agents(handles[0], method="random", n=self.n_agents)
        self.env.add_agents(handles[1], method="random", n=self.n_agents + self.args.more_enemy)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            num_agents = self.env.get_num(handles[0])
            fixed_num_agents = self.env.get_num(handles[1])
            if num_agents < self.n_agents:
                self.env.add_agents(handles[0], method="random", n=self.n_agents - num_agents)
            # if fixed_num_agents < self.n_agents:
            #     self.env.add_agents(handles[1], method="random", n=self.n_agents - fixed_num_agents)

            obs_all = self.env.get_observation(handles[0])
            fixed_obs_all = self.env.get_observation(handles[1])
            view = obs_all[0]
            feature = obs_all[1]
            fixed_view = fixed_obs_all[0]
            fixed_feature = fixed_obs_all[1]
            obs = []
            fixed_obs = []
            state = self.env.get_global_minimap(self.args.mini_map_shape, self.args.mini_map_shape).flatten()

            for j in range(self.n_agents):
                obs.append(np.concatenate([view[j].flatten(), feature[j]]))
                fixed_obs.append(np.concatenate([fixed_view[j].flatten(), fixed_feature[j]]))

            # obs = self.env.get_obs()
            # state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_action = np.ones(self.n_actions)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                if isinstance(action, np.int64):
                    action = action.astype(np.int32)
                else:
                    action = action.cpu()
                    action = action.numpy().astype(np.int32)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # reward, terminated, info = self.env.step(actions)
            acts = [[], []]
            acts[0] = np.array(actions)
            # print(actions)
            acts[1] = np.array(np.random.randint(0, self.args.fixed_n_actions, size=self.env.get_num(handles[1]), dtype='int32'))
            self.env.set_action(handles[0], acts[0])
            self.env.set_action(handles[1], acts[1])
            terminated = self.env.step()
            reward = sum(self.env.get_reward(handles[0]))
            fixed_reward = sum(self.env.get_reward(handles[1]))
            self.env.clear_dead()
            if step == self.episode_limit - 1:
                terminated = 1.

            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        # obs = self.env.get_obs()
        # state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_action = np.ones(self.n_actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy())
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step
