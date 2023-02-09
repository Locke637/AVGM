import argparse
import random

from s2clientprotocol.raw_pb2 import Enemy
"""
Here are the param for the training

"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='5m_vs_6m', help='the map of the game')
    # 2z3c 8m corridor 5m_vs_6m 8m_vs_9m 6h_vs_8z 3s_vs_4z 10m_vs_11m 27m_vs_30m
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999), help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='ours', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=5000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=1, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=1, help='number of the epoch to evaluate the agent')  # 7
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_mixer_args_magent(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 700000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.n_epoch = 10001
    args.train_steps = 1

    # experience replay
    args.batch_size = 1  # 4
    args.buffer_size = int(1e3)  # 3e2 others:1e3

    # how often to save the model
    args.save_cycle = 1000  # 1000
    args.evaluate_epoch = 1
    args.evaluate_cycle = 1

    # how often to update the target_net #200
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    args.render = False
    args.use_fixed_model = False
    args.use_target_actions = False
    args.use_pilike_coll = False
    # args.enemy_num = enemy_num[args.map]
    args.lambda_mi = 0.001
    args.noise_dim = 32
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    # pursuit 100000 3m 10000 others 700000
    anneal_steps = 700000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.n_epoch = 5001
    args.train_steps = 1

    # experience replay
    args.batch_size = 4  # 4
    args.buffer_size = int(1e3)  # 3e2 others:1e3

    # how often to save the model
    args.save_cycle = 10000  # 1000

    # how often to update the target_net #200
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    # if 'wvdn' in args.alg:
    #     args.noise_dim = 16
    #     args.id_dim = 0
    # else:

    #     args.id_dim = 0
    args.render = False
    args.use_fixed_model = False
    args.use_target_actions = False
    args.use_pilike_coll = False
    # args.other_feature_dim = 5
    args.enemy_num = enemy_num[args.map]
    args.lambda_mi = 0.001
    args.noise_dim = 16
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 1  # 0.5
    args.anneal_epsilon = 0.000064
    args.min_epsilon = 0.05  # 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.n_epoch = 5001
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 1.0
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.05
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
