import numpy as np
import os
# import wandb
from numpy.core.fromnumeric import mean
# from common.rollout import RolloutWorker, CommRolloutWorker
from common.rollout_magent import RolloutWorkerMagent, CommRolloutWorkerMagent
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from renderer import PyGameRenderer
from renderer.server import BattleServer as Server
import time
import pickle

class RunnerMagent:
    def __init__(self, env, args):
        self.env = env
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorkerMagent(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorkerMagent(env, self.agents, args)
        if args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        # self.win_rates = []
        self.episode_rewards = []
        self.plt_z_lists = []

        # # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # self.save_path = self.args.result_dir + '/' + args.map + '/'
        # if not os.path.exists(self.save_path):
        #     os.makedirs(self.save_path)
        # self.file_name = self.save_path + str(args.env_name) + '_' + str(args.n_agents) + '_' + str(
        #     args.map_size) + '_' + args.name_time

    def run(self, num):
        train_steps = 0
        episode_rewards = 0
        fixed_rewards = 0
        plot_rewards = []
        st = time.time()
        for epoch in range(self.args.n_epoch):
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                if 'ours' in self.args.alg:
                    episode, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma_obs_wvdn(episode_idx)
                else:
                    episode, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma(episode_idx)
                episodes.append(episode)
                episode_rewards += episode_reward
                fixed_rewards += fixed_reward
                # plot_rewards.append(episode_reward)
                if epoch % self.args.evaluate_cycle == 0:
                    epr, fr = self.evaluate()
                    t = time.time() - st
                    st = time.time()
                    print('train epoch {}, reward {}, time {}, rate {}'.format(epoch, [epr, fr], t, rate))
                    episode_rewards = 0
                    fixed_rewards = 0
                    self.episode_rewards.append(epr)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            elif not self.args.load_model:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

    def evaluate(self):
        episode_rewards = 0
        fr = 0
        if self.args.use_random_num:
            var_num = self.args.n_agents + np.random.randint(-self.args.random_num, self.args.random_num + 1)
        else:
            var_num = self.args.n_agents
        for episode_idx in range(1):
            if self.args.alg == 'ours':
                _, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma_obs_wvdn(episode_idx, evaluate=True)
            elif self.args.alg == 'central_v+g2anet':
                _, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma(episode_idx, evaluate=True)
            elif 'ours' in self.args.alg:
                if self.args.use_random_num:
                    _, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma_obs_wvdn_test(episode_idx, evaluate=True, var_num=var_num)
                else:
                    _, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma_obs_wvdn(episode_idx, evaluate=True)
            else:
                _, episode_reward, rate, fixed_reward = self.rolloutWorker.generate_episode_ma_test(episode_idx, evaluate=True, var_num=var_num)
            # print(episode_reward, var_num)
            episode_rewards += round(episode_reward / var_num, 2)
            fr += round(fixed_reward / var_num, 2)
        return episode_rewards, fr

    def plt(self):
        # plt.figure(figsize=(20, 10))
        # plt.ylim([0, 105])
        # plt.cla()
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('win_rates')
        #
        # plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(self.args.run_time), format='png')
        # np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(self.args.run_time), self.episode_rewards)
        plt.close()

    # def render(self):
    #     PyGameRenderer().start(Server(self.args, self.agents), animation_total=10)
