from runner import RunnerMagent
import magent
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args, get_mixer_args_magent


def get_config_pursuit_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type("predator", {'width': 1, 'length': 1, 'hp': 1, 'speed': 1, 'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1), 'attack_penalty': -0.15})

    prey = cfg.register_agent_type("prey", {'width': 1, 'length': 1, 'hp': 1, 'speed': 0, 'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)})

    predator_group = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(prey_group, index='any')

    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[0.5, 0.5])

    return cfg


if __name__ == '__main__':
    view_dic = {'pursuit': 5}
    num_neighbor_dic = {'pursuit': 2}
    for i in range(1):
        args = get_common_args()
        args.alg = 'ours_mcmc'  # vdn qmix qtrans ours_mcmc iql central_v+g2anet maven
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args_magent(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        args.map_size = 6  # 80 30
        args.env_name = 'pursuit'  # double_attack battle pursuit
        env = magent.GridWorld(get_config_pursuit_attack(args.map_size))
        args.n_agents = 3  # 6
        args.more_walls = 0
        args.more_enemy = 0
        args.random_num = 1
        args.mini_map_shape = 6  # battle:20 pursuit:30
        args.run_time = '0'

        handles = env.get_handles()
        env.set_seed(args.seed)
        args.map = args.env_name
        eval_obs = None
        feature_dim = env.get_feature_space(handles[0])
        view_dim = env.get_view_space(handles[0])
        real_view_shape = view_dim
        v_dim_total = view_dim[0] * view_dim[1] * view_dim[2]
        obs_shape = (v_dim_total + feature_dim[0], )
        args.n_actions = env.action_space[0][0]
        args.fixed_n_actions = env.action_space[1][0]
        args.state_shape = (args.mini_map_shape * args.mini_map_shape) * 2
        args.view_shape = v_dim_total
        args.act_dim = env.action_space[0][0]
        args.feature_shape = feature_dim[0]
        args.real_view_shape = real_view_shape
        args.obs_shape = obs_shape[0]
        args.episode_limit = 100
        args.view_field = view_dic[args.env_name]
        args.num_neighbor = num_neighbor_dic[args.env_name]
        args.enemy_feats_dim = 0
        args.pos_dim = 2
        args.use_fixed_model = False
        args.eva = False
        args.use_random_num = False
        args.use_mixing = True
        args.use_other_feature = True
        args.coding = False
        
        runner = RunnerMagent(env, args)
        runner.run(i)
