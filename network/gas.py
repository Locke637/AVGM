# from pygame.mixer import pre_init
# from builtins import print
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import one_hot_categorical


class VarDistribution(nn.Module):
    def __init__(self, input_shape, args):
        super(VarDistribution, self).__init__()
        self.args = args

        # self.GRU = nn.GRU(args.state_shape, 64)
        self.fc_0 = nn.Linear(input_shape, 64)

        self.fc_1 = nn.Linear(64, 32)
        self.fc_2 = nn.Linear(32, args.noise_dim)

    def forward(self, inputs):  # q_value.
        # get sigma(q) by softmax
        # print(inputs.shape)
        # tx, h = self.GRU(inputs)  # (1, 1, 64)
        # print(tx.shape)
        # print(h.shape)
        h = f.relu(self.fc_0(inputs))
        # x = f.relu(self.fc_1(h.squeeze(0)))
        x = f.relu(self.fc_1(h))
        x = self.fc_2(x)
        output = f.softmax(x, dim=-1)
        return output


class MLP_padding(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, output_shape, args):
        super(MLP_padding, self).__init__()
        self.args = args
        self.input_shape = input_shape[0]
        self.other_input_shape = input_shape[1]
        self.ll = args.n_agents - 1
        self.cate_dim = 32

        self.fc11 = nn.Linear(input_shape[0], 64)
        # self.fc12 = nn.Linear(input_shape[1], self.cate_dim)

        self.fc120 = nn.Linear(input_shape[1], 32)
        self.fc121 = nn.Linear(32, self.cate_dim)

        # self.fc1 = nn.Linear(input_shape[0] + (args.n_agents - 1) * input_shape[1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + self.cate_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim + input_shape[1], output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim + input_shape[1], 1)

    def forward(self, inputs, other_inputs, use_test=False, ret_cate=False, act_idx=None):
        input_x = self.fc11(inputs)

        if use_test:
            final_q = -99999
            test_inputs = torch.eye(self.cate_dim).cuda()
            var_q = []
            for t in test_inputs:
                x = f.relu(torch.cat((input_x, t), dim=0))
                h = f.relu(self.fc2(x))
                q = self.fc3(h)
                var_q.append(q[act_idx])
                if q[act_idx] > final_q:
                    ret_q = q
                    final_q = q[act_idx]
                # if torch.max(q) > final_q:
                #     ret_q = q
                #     final_q = torch.max(q)
            # print(ret_q, final_q)
            var_q = torch.stack(var_q, dim=0)
            var = torch.var(var_q, dim=0)
            # print(var_q, var)
            q = ret_q
        else:
            other_inputs = torch.stack(other_inputs, dim=0).reshape(-1)

            # input_others = self.fc12(other_inputs)

            input_others = f.relu(self.fc120(other_inputs))
            input_others = self.fc121(input_others)

            z_prob = f.softmax(input_others, dim=0)
            samples = torch.zeros(self.cate_dim).cuda()
            samples_idx = torch.argmax(z_prob)
            samples[samples_idx] = 1
            input_others_one_hot = samples + (z_prob - z_prob.detach())
            # print(input_others_one_hot)

            x = f.relu(torch.cat((input_x, input_others_one_hot), dim=0))
            h = f.relu(self.fc2(x))
            q = self.fc3(h)
        if ret_cate:
            return q, input_others_one_hot
        elif use_test:
            return q, var
        else:
            return q


class MLP_GAS_bup(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, output_shape, args):
        super(MLP_GAS_bup, self).__init__()
        self.args = args
        self.input_shape = input_shape[0]
        self.other_input_shape = input_shape[1]
        self.ll = args.n_agents - 1
        self.cate_dim = 32

        self.fc11 = nn.Linear(input_shape[0], 64)
        self.fc12 = nn.Linear(input_shape[1], self.cate_dim)
        self.fc1q = nn.Linear(input_shape[0], 64)

        self.attn = nn.Linear(args.rnn_hidden_dim + self.cate_dim, 1)
        self.attn_out = nn.Linear(self.cate_dim, self.cate_dim)
        # self.fc120 = nn.Linear(input_shape[1], 32)
        # self.fc121 = nn.Linear(32, self.cate_dim)

        # self.fc1 = nn.Linear(input_shape[0] + (args.n_agents - 1) * input_shape[1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + self.cate_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim + input_shape[1], output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim + input_shape[1], 1)

    def forward(self, inputs, other_inputs, use_test=False, ret_cate=False, act_idx=None):
        input_x = self.fc11(inputs)
        input_q = self.fc1q(inputs)

        if use_test:
            final_q = -99999
            test_inputs = torch.eye(self.cate_dim).cuda()
            var_q = []
            var = 0
            for t in test_inputs:
                x = f.relu(torch.cat((input_x, t), dim=0))
                h = f.relu(self.fc2(x))
                q = self.fc3(h)
                # print(torch.argmax(q), torch.argmax(t))
                var_q.append(q[act_idx])
                if q[act_idx] > final_q:
                    ret_q = q
                    final_q = q[act_idx]
            # var_q = torch.stack(var_q, dim=0)
            # var = torch.var(var_q, dim=0)
            # # print(var_q, var)
            q = ret_q
        else:
            input_others_mixing = []
            input_others_mixing_v = []
            weights = []
            for oi in other_inputs:
                emb = self.fc12(oi)
                input_others_mixing.append(emb)
                input_others_mixing_v.append(f.relu(emb))

            for i in range(len(input_others_mixing)):
                weights.append(self.attn(f.relu(torch.cat((input_q, input_others_mixing[i]), dim=0))))
            weights = torch.stack(weights, dim=0)
            normalized_weights = f.softmax(weights, dim=0)
            input_others_mixing_v = torch.stack(input_others_mixing_v, dim=0)
            input_others = torch.mm(normalized_weights.reshape(1, -1), input_others_mixing_v)
            input_others = self.attn_out(f.relu(input_others))

            # input_others_mixing = torch.stack(input_others_mixing, dim=0)
            # # print(input_others_mixing)
            # input_others = torch.mean(input_others_mixing, dim=0)
            # # print(input_others)

            z_prob = f.softmax(input_others.reshape(-1), dim=0)
            samples = torch.zeros(self.cate_dim).cuda()
            samples_idx = torch.argmax(z_prob)
            samples[samples_idx] = 1
            # print(samples)
            input_others_one_hot = samples + (z_prob - z_prob.detach())
            # print(input_others_one_hot)

            x = f.relu(torch.cat((input_x, input_others_one_hot.squeeze(0)), dim=0))
            h = f.relu(self.fc2(x))
            q = self.fc3(h)
        if ret_cate:
            return q, input_others_one_hot
        elif use_test:
            return q, var
        else:
            return q

class ACT_RE(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(ACT_RE, self).__init__()
        self.args = args
        self.input_shape = input_shape[0]
        self.other_input_shape = input_shape[1]
        self.ll = args.n_agents - 1
        self.cate_dim = 32

        self.fc11 = nn.Linear(input_shape[0] + output_shape, 64)
        self.fc12 = nn.Linear(input_shape[1], self.cate_dim)

        self.attn = nn.Linear(args.rnn_hidden_dim + self.cate_dim, 1)
        self.attn_out = nn.Linear(self.cate_dim, self.cate_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim + self.cate_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs, other_inputs, act_idx):
        inputs = torch.cat((inputs, act_idx), dim=0)
        input_x = self.fc11(inputs)
        input_others_mixing = []
        input_others_mixing_v = []
        weights = []
        for oi in other_inputs:
            emb = self.fc12(oi)
            input_others_mixing.append(emb)
            input_others_mixing_v.append(emb)
            # input_others_mixing_v.append(f.relu(emb))

        for i in range(len(input_others_mixing)):
            weights.append(self.attn(f.relu(torch.cat((input_x, input_others_mixing[i]), dim=0))))
        weights = torch.stack(weights, dim=0)
        normalized_weights = f.softmax(weights, dim=0)
        input_others_mixing_v = torch.stack(input_others_mixing_v, dim=0)
        input_others = torch.mm(normalized_weights.reshape(1, -1), input_others_mixing_v)
        input_others = self.attn_out(f.relu(input_others))

        # input_others_mixing = torch.stack(input_others_mixing, dim=0)
        # input_others = torch.mean(input_others_mixing, dim=0)
        # input_others = self.attn_out(f.relu(input_others))

        # # print(input_others)
        z_prob = f.softmax(input_others.reshape(-1), dim=0)

        # input_others_one_hot = one_hot_categorical.OneHotCategoricalStraightThrough(z_prob).rsample()
        # samples = input_others_one_hot.detach()

        samples = torch.zeros(self.cate_dim).cuda()
        samples_idx = torch.argmax(z_prob)
        samples[samples_idx] = 1
        # print(samples)
        input_others_one_hot = samples + (z_prob - z_prob.detach())
        # print(input_others_one_hot)

        x = f.relu(torch.cat((input_x, input_others_one_hot.squeeze(0)), dim=0))
        h = f.relu(self.fc2(x))
        r = self.fc3(h)
        return r, samples

class MLP_GAS(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, output_shape, args):
        super(MLP_GAS, self).__init__()
        self.args = args
        self.input_shape = input_shape[0]
        self.other_input_shape = input_shape[1]
        self.ll = args.n_agents - 1
        self.cate_dim = 32

        self.fc11 = nn.Linear(input_shape[0], 64)
        # self.fc12 = nn.Linear(input_shape[1], self.cate_dim)

        # self.fc1 = nn.Linear(input_shape[0] + (args.n_agents - 1) * input_shape[1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + self.cate_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim + input_shape[1], output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim + input_shape[1], 1)

    def forward(self, inputs, other_inputs, use_test=False, act_idx=None):
        input_x = self.fc11(inputs)
        if use_test:
            final_q = -99999
            test_inputs = torch.eye(self.cate_dim).cuda()
            var_q = []
            var = 0
            for t in test_inputs:
                x = f.relu(torch.cat((input_x, t), dim=0))
                h = f.relu(self.fc2(x))
                q = self.fc3(h)
                # print(torch.argmax(q), torch.argmax(t))
                var_q.append(q[act_idx])
                if q[act_idx] > final_q:
                    ret_q = q
                    final_q = q[act_idx]
            # var_q = torch.stack(var_q, dim=0)
            # var = torch.var(var_q, dim=0)
            # # print(var_q, var)
            q = ret_q
        else:
            input_others_one_hot = other_inputs
            # print(input_others_one_hot)))
            x = f.relu(torch.cat((input_x, input_others_one_hot.squeeze(0)), dim=0))
            h = f.relu(self.fc2(x))
            q = self.fc3(h)
        if use_test:
            return q, var
        else:
            return q