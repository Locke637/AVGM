from numpy.core.fromnumeric import ptp
from pygame.mixer import pre_init
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
# from network.our_net import HierarchicalPolicy, BootstrappedRNN, VarDistribution, HierarchicalPolicyState, \
#     VarDistributionState
from network.qmix_net import QMixNet
from network.base_net import MLP
from network.gas import MLP_GAS, ACT_RE
from network.vdn_net import VDNNet, ShapleyVDNNet
import numpy as np
# import time
# import random

# torch.cuda.set_device(0)


class OURS_MC:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.other_feature_dim = self.args.pos_dim
        self.plot_w = []

        # input shaoe of rnn
        input_shape = self.obs_shape
        if self.args.use_other_feature:
            coll_input_shape = [self.obs_shape, self.other_feature_dim + self.n_actions]
        else:
            coll_input_shape = [self.obs_shape, self.n_actions]
        coll_output_shape = self.n_actions
        cost_input_shape = self.obs_shape
        cost_output_shape = self.n_actions
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn_coll = MLP_GAS(coll_input_shape, coll_output_shape, args)
        self.target_rnn_coll = MLP_GAS(coll_input_shape, coll_output_shape, args)
        self.action_encoder = ACT_RE(coll_input_shape, coll_output_shape, args)
        self.eval_rnn_cost = MLP(cost_input_shape, cost_output_shape, args)
        self.target_rnn_cost = MLP(cost_input_shape, cost_output_shape, args)
        self.eval_rnn_policy = MLP(input_shape, args.n_actions, args)
        self.target_rnn_policy = MLP(input_shape, args.n_actions, args)
        # self.mi_net = VarDistribution(input_shape * 2, args)
        self.mse_loss = torch.nn.MSELoss()

        if self.args.use_mixing:
            self.eval_qmix_net = QMixNet(args)  # mix the q value
            self.target_qmix_net = QMixNet(args)
        else:
            self.eval_qmix_net = VDNNet()
            self.target_qmix_net = VDNNet()

        if self.args.cuda:
            self.eval_rnn_coll.cuda()
            self.eval_rnn_cost.cuda()
            self.eval_rnn_policy.cuda()
            self.target_rnn_coll.cuda()
            self.target_rnn_cost.cuda()
            self.target_rnn_policy.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.action_encoder.cuda()
            # self.mi_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + args.run_time
        # 如果存在模型则加载模型
        if self.args.load_model:
            # print('yoooooo')
            if os.path.exists(self.model_dir + '/' + str(self.args.load_num) + '_rnn_net_params.pkl'):
                # path_z_policy = self.model_dir + '/z_policy_params.pkl'
                # path_rnn = self.model_dir + '/rnn_net_params.pkl'
                # path_qmix = self.model_dir + '/qmix_net_params.pkl'
                # path_mi = self.model_dir + '/mi_net_params.pkl'
                path_rnn = self.model_dir + '/' + str(self.args.load_num) + '_rnn_net_params.pkl'
                path_coll = self.model_dir + '/' + str(self.args.load_num) + '_coll_net_params.pkl'
                path_cost = self.model_dir + '/' + str(self.args.load_num) + '_cost_net_params.pkl'
                path_qmix = self.model_dir + '/' + str(self.args.load_num) + '_qmix_net_params.pkl'
                path_act_re = self.model_dir + '/' + str(self.args.load_num) + '_act_re_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn_policy.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_rnn_coll.load_state_dict(torch.load(path_coll, map_location=map_location))
                self.eval_rnn_cost.load_state_dict(torch.load(path_cost, map_location=map_location))
                self.action_encoder.load_state_dict(torch.load(path_act_re, map_location=map_location))
                # self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                # torch.save(self.eval_rnn_policy.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
                # torch.save(self.eval_rnn_coll.state_dict(), self.model_dir + '/' + num + '_coll_net_params.pkl')
                # torch.save(self.eval_rnn_cost.state_dict(), self.model_dir + '/' + num + '_cost_net_params.pkl')
                print('Successfully load the model: {}, {}, {} and {}'.format(path_coll, path_rnn, path_qmix, path_cost))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn_coll.load_state_dict(self.eval_rnn_coll.state_dict())
        self.target_rnn_cost.load_state_dict(self.eval_rnn_cost.state_dict())
        self.target_rnn_policy.load_state_dict(self.eval_rnn_policy.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + \
                               list(self.eval_rnn_coll.parameters()) + list(self.eval_rnn_cost.parameters())
        self.action_encoder_parameters = list(self.action_encoder.parameters())
        self.p_eval_parameters = list(self.eval_rnn_policy.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            self.p_optimizer = torch.optim.RMSprop(self.p_eval_parameters, lr=args.lr)
            self.action_encoder_optimizer = torch.optim.RMSprop(self.action_encoder_parameters, lr=args.lr)
        print('Init alg OURS_MCMC')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        self.plot_w = []
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            elif 'neighbor' not in key:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], batch['r'], \
                                                                batch['avail_u'], batch['avail_u_next'], \
                                                                batch['terminated']

        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        # q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        q_critic, q_critic_next, q_values_policy, q_values_policy_target, act_re_loss = self.get_q_values_critic_mc(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        if self.args.use_mixing:
            q_total_eval = self.eval_qmix_net(q_critic, s)
            q_total_target = self.target_qmix_net(q_critic_next, s_next)
        else:
            q_total_eval = self.eval_qmix_net(q_critic)
            q_total_target = self.target_qmix_net(q_critic_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        if self.args.use_mixing:
            td_error = (q_total_eval - targets.detach())
        else:
            td_error = (q_total_eval.squeeze(-1) - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
        ql_loss = (masked_td_error**2).sum() / mask.sum()
        self.optimizer.zero_grad()
        ql_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        policy_td_errror = (q_values_policy.squeeze(-1) - q_values_policy_target.detach())
        policy_masked_td_error = mask * policy_td_errror.cuda()
        policy_ql_loss = (policy_masked_td_error**2).sum() / mask.sum()
        self.p_optimizer.zero_grad()
        policy_ql_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_eval_parameters, self.args.grad_norm_clip)
        self.p_optimizer.step()

        act_re_logit = act_re_loss[0].cuda()
        act_re_label = act_re_loss[1].cuda()
        action_represention_loss = self.mse_loss(act_re_logit, act_re_label)
        # print(action_represention_loss)
        self.action_encoder_optimizer.zero_grad()
        action_represention_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_encoder_parameters, self.args.grad_norm_clip)
        self.action_encoder_optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn_coll.load_state_dict(self.eval_rnn_coll.state_dict())
            self.target_rnn_cost.load_state_dict(self.eval_rnn_cost.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def find_policy_q_v3(self, inputs, other_feature, action, action_cost, other_avail_act):
        search_acts_tmp = torch.eye(self.n_actions).cuda()
        search_acts = search_acts_tmp
        # for ti in range(self.n_actions):
        #     if other_avail_act[ti] == 1:
        #         search_acts.append(search_acts_tmp[ti])
        # print(search_acts)
        q_value_final = -9999999
        q_value_compare = -9999999
        # weight_final = -9999999
        for a in search_acts:
            # other_feature = torch.cat((other_feature, a), dim=0)
            # inputs_cost = inputs_0.unsqueeze(0).cuda()
            # print(inputs_tmp, inputs)
            others_inputs = [a]
            # if self.args.cuda:
            #     inputs = inputs.cuda()
            # others_inputs = others_inputs.cuda()
            q_value_tmp = self.eval_rnn_coll(inputs, others_inputs)
            # q_value_tmp, weight_tmp = self.target_rnn_coll(inputs)
            # print(action, action_cost)
            q_value = torch.gather(q_value_tmp, dim=0, index=action)
            q_value_cost = torch.gather(q_value_tmp, dim=0, index=action_cost)
            # print(q_value, q_value_cost)
            # print(torch.exp(weight_tmp) *  q_value, weight_tmp, q_value)
            # if torch.exp(weight_tmp) * q_value > q_value_compare:
            #     q_value_final = q_value - 1.0 * q_value_cost
            #     weight_final = weight_tmp
            #     q_value_cost_final = q_value_cost
            #     q_value_compare = torch.exp(weight_tmp) * q_value
            if q_value > q_value_compare:
                q_value_final = q_value - 1.0 * q_value_cost
                # weight_final = weight_tmp
                q_value_cost_final = q_value_cost
                q_value_compare = q_value
            # if weight_tmp > weight_final:
            #     q_value_final = q_value - 1.0 * q_value_cost
            #     weight_final = weight_tmp
            #     q_value_cost_final = q_value_cost
            #     q_value_compare = q_value
            # print('q', q_value_final)
        # print(weights, weights * q_value_cost_final)
        # q_value_final -= 0.01 * q_value_cost_final
        # q_value_final -= weights * q_value_cost_final
        return q_value_final

    def get_q_values_critic_mc(self, batch, max_episode_len):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot, u, avail_u = batch['o'], batch['o_next'], batch['u_onehot'], batch['u'], batch['avail_u']
        if self.args.cuda:
            obs = obs.cuda()
            u_onehot = u_onehot.cuda()
            u = u.cuda()
        neighbor = batch['neighbor']
        neighbor_pos = batch['neighbor_pos']
        rlabel = batch['r']
        q_critic = []
        q_critic_next = []
        q_values_policy_target = []
        q_values_policy_eval = []
        dq_loss = 0
        act_re_loss = []
        act_re_loss_label = []
        for nb in range(len(batch['o'])):
            q_values_critic_nb = []
            q_values_critic_nb_next = []
            q_values_policy_nb_eval = []
            q_values_policy_nb_target = []
            act_re_list = {}
            for transition_idx in range(max_episode_len):
                act_re_list[transition_idx] = {}
                pr_tot = 0
                for i in range(self.n_agents):
                    ids = neighbor[nb][transition_idx][i]
                    if ids:
                        tmp = []
                        for pos_id, n_i in enumerate(ids):
                            other_onehot_act = u_onehot[nb][transition_idx][n_i]
                            if self.args.use_other_feature:
                                other_feature = torch.tensor(neighbor_pos[nb][transition_idx][i][pos_id], dtype=torch.float32).cuda()
                                tmp.append(torch.cat((other_feature, other_onehot_act), dim=0))
                            else:
                                tmp.append(other_onehot_act)
                        pr, act_re = self.action_encoder(obs[nb][transition_idx][i], tmp, u_onehot[nb][transition_idx][i])
                        act_re_list[transition_idx][i] = act_re
                    else:
                        tmp = []
                        other_onehot_act = torch.zeros(self.n_actions).cuda()
                        if self.args.use_other_feature:
                            other_feature = torch.zeros(self.other_feature_dim).cuda()
                            tmp.append(torch.cat((other_feature, other_onehot_act), dim=0))
                        else:
                            tmp.append(other_onehot_act)
                        pr, act_re = self.action_encoder(obs[nb][transition_idx][i], tmp, u_onehot[nb][transition_idx][i])
                    pr_tot += pr
                act_re_loss.append(pr_tot)
                act_re_loss_label.append(rlabel[nb][transition_idx])
            for transition_idx in range(max_episode_len):
                q_values_transition = []
                q_values_transition_next = []
                q_values_transition_policy = []
                q_values_transition_policy_eval = []
                transition_idx_target = min(transition_idx + 1, max_episode_len - 1)
                roll_ids = {}
                q_value_policy_eval_list = {}
                for i in range(self.n_agents):
                    act_idx = u[nb][transition_idx][i]
                    q_value_policy_eval_find = self.eval_rnn_policy(obs[nb][transition_idx][i])
                    q_value_policy_eval = torch.gather(q_value_policy_eval_find, dim=0, index=act_idx)
                    q_value_policy_eval_list[i] = q_value_policy_eval
                for i in range(self.n_agents):
                    roll_ids[i] = []
                    act_idx = u[nb][transition_idx][i]
                    q_value_policy_eval = q_value_policy_eval_list[i]
                    ids = neighbor[nb][transition_idx][i]
                    ids_next = neighbor[nb][transition_idx_target][i]
                    if ids:
                        q_cost = self.eval_rnn_cost(obs[nb][transition_idx][i])
                        avail_act = avail_u[nb][transition_idx][i]
                        q_cost[avail_act == 0.0] = -9999999
                        action_cost = torch.argmax(q_cost).unsqueeze(-1)
                        q_policy_tmp = 0
                        q_eval_mean = 0
                        act_re = act_re_list[transition_idx][i]
                        q_eval = self.eval_rnn_coll(obs[nb][transition_idx][i], act_re)
                        q_eval_act = torch.gather(q_eval, dim=0, index=act_idx)
                        find_input = []
                        q_eval_find, var_q = self.eval_rnn_coll(obs[nb][transition_idx][i], find_input, use_test=True, act_idx=act_idx)
                        q_pi_tmp = torch.gather(q_eval_find, dim=0, index=act_idx) - 1.0 * torch.gather(q_eval_find, dim=0, index=action_cost)
                        q_eval_mean = q_eval_act
                        q_policy_tmp = q_pi_tmp
                        q_policy_tmp = q_policy_tmp.squeeze(-1)
                    else:
                        cost_act_idx = u[nb][transition_idx][i]
                        q_eval = self.eval_rnn_cost(obs[nb][transition_idx][i])
                        q_eval = torch.gather(q_eval, dim=0, index=cost_act_idx)
                        q_eval_mean = q_eval
                        q_policy_tmp = q_eval.clone().squeeze(-1)
                    q_values_transition.append(q_eval_mean)
                    q_values_transition_policy.append(q_policy_tmp)
                    q_values_transition_policy_eval.append(q_value_policy_eval)
                    if ids_next:
                        q_target_mean = 0
                        q_target_find = 0
                        tmp_next = act_re_list[transition_idx_target][i]
                        q_target = self.target_rnn_coll(obs[nb][transition_idx_target][i], tmp_next)
                        q_target_find = q_target
                        avail_act_next = avail_u[nb][transition_idx_target][i]
                        q_target_find[avail_act_next == 0.0] = -9999999
                        q_target_mean = max(q_target_find)
                        q_values_transition_next.append(q_target_mean)
                    else:
                        q_target = self.target_rnn_cost(obs[nb][transition_idx_target][i])
                        avail_act_next = avail_u[nb][transition_idx_target][i]
                        q_target[avail_act_next == 0.0] = -9999999
                        q_target = max(q_target)
                        q_values_transition_next.append(q_target)
                q_values_transition = torch.stack(q_values_transition, dim=0)
                q_values_transition_next = torch.stack(q_values_transition_next, dim=0)
                q_values_transition_policy = torch.stack(q_values_transition_policy, dim=0)
                q_values_transition_policy_eval = torch.stack(q_values_transition_policy_eval, dim=0)
                q_values_critic_nb.append(q_values_transition)
                q_values_critic_nb_next.append(q_values_transition_next)
                q_values_policy_nb_target.append(q_values_transition_policy)
                q_values_policy_nb_eval.append(q_values_transition_policy_eval)
            q_values_critic_nb = torch.stack(q_values_critic_nb, dim=0).cuda()
            q_values_critic_nb_next = torch.stack(q_values_critic_nb_next, dim=0).cuda()
            q_values_policy_nb_target = torch.stack(q_values_policy_nb_target, dim=0).cuda()
            q_values_policy_nb_eval = torch.stack(q_values_policy_nb_eval, dim=0).cuda()
            q_critic.append(q_values_critic_nb)
            q_critic_next.append(q_values_critic_nb_next)
            q_values_policy_target.append(q_values_policy_nb_target)
            q_values_policy_eval.append(q_values_policy_nb_eval)
        q_values_policy_eval_ret = torch.stack(q_values_policy_eval, dim=0)
        q_values_policy_target_ret = torch.stack(q_values_policy_target, dim=0)
        q_critic = torch.stack(q_critic, dim=0)
        q_critic_next = torch.stack(q_critic_next, dim=0)
        act_re_loss = torch.stack(act_re_loss, dim=0)
        act_re_loss_label = torch.stack(act_re_loss_label, dim=0)
        return q_critic, q_critic_next, q_values_policy_eval_ret, q_values_policy_target_ret, [act_re_loss, act_re_loss_label]

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden_coll = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden_coll = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_hidden_cost = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden_cost = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_hidden_policy = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden_policy = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        print('save model!')
        num = str(train_step // self.args.save_cycle)
        # num = str(train_step // 10)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # torch.save(self.z_policy.state_dict(), self.model_dir + '/' + num + '_z_policy_params.pkl')
        # torch.save(self.mi_net.state_dict(), self.model_dir + '/' + num + '_mi_net_params.pkl')
        # torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn_policy.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_rnn_coll.state_dict(), self.model_dir + '/' + num + '_coll_net_params.pkl')
        torch.save(self.eval_rnn_cost.state_dict(), self.model_dir + '/' + num + '_cost_net_params.pkl')
        torch.save(self.action_encoder.state_dict(), self.model_dir + '/' + num + '_act_re_net_params.pkl')
