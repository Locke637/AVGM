from pygame.mixer import pre_init
from builtins import print
import torch
import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class MLP(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, output_shape, args):
        super(MLP, self).__init__()
        self.args = args
        # self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        # h = torch.cat((x, feature), dim=0)
        # print(self.fc1(obs))
        h = f.relu(self.fc2(x))
        q = self.fc3(h)
        # print(q)
        return q


class MLP_MC(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, output_shape, args):
        super(MLP_MC, self).__init__()
        self.args = args
        self.input_shape = input_shape[0]

        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + input_shape[1], args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, output_shape)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim + input_shape[1], output_shape)
        # self.fc4 = nn.Linear(args.rnn_hidden_dim + input_shape[1], 1)

    def forward(self, inputs):
        obs = inputs[:self.input_shape]
        act = inputs[self.input_shape:]
        # print(obs, act)
        x = f.relu(self.fc1(obs))
        # h = torch.cat((x, feature), dim=0)
        # print(self.fc1(obs))
        x = torch.cat((x, act), dim=0)
        h = f.relu(self.fc2(x))
        # print(h)
        # h = torch.cat((h, act), dim=0)
        # print(h)
        q = self.fc3(h)
        w = self.fc4(h)
        # print(q)
        return q, w

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
        self.fc12 = nn.Linear(input_shape[1], self.cate_dim)
        
        self.attn = nn.Linear(args.rnn_hidden_dim + self.cate_dim, 1)
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

        if use_test:
            final_q = -99999
            test_inputs = torch.eye(self.cate_dim).cuda()
            for t in test_inputs:
                x = f.relu(torch.cat((input_x, t), dim=0))
                h = f.relu(self.fc2(x))
                q = self.fc3(h)
                # print(torch.argmax(q), torch.argmax(t))
                if q[act_idx] > final_q:
                    ret_q = q
                    final_q = q[act_idx]
            q = ret_q
        else:
            input_others_mixing = []
            weights = []
            for oi in other_inputs:
                input_others_mixing.append(self.fc12(oi))

            # for i in range(len(input_others_mixing)):
            #     weights.append(self.attn(f.relu(torch.cat((input_x, input_others_mixing[i]), dim=0))))
            # weights = torch.stack(weights, dim=0)
            # normalized_weights = f.softmax(weights, dim=0)

            input_others_mixing = torch.stack(input_others_mixing, dim=0)
            normalized_weights = torch.ones(len(other_inputs)).unsqueeze(0).cuda()
            # print(input_others_mixing, normalized_weights)
            input_others = torch.mm(normalized_weights.reshape(1, -1), input_others_mixing)
            # print(input_others)

            max_idx = torch.argmax(input_others)
            input_others_one_hot = torch.zeros(self.cate_dim).cuda()
            input_others_one_hot[max_idx] = 1

            x = f.relu(torch.cat((input_x, input_others_one_hot), dim=0))
            h = f.relu(self.fc2(x))
            q = self.fc3(h)
        if ret_cate:
            return q, input_others_one_hot
        else:
            return q


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
