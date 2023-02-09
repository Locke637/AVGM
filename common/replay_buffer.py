import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        if 'ours' in self.args.alg:
            self.buffers['neighbor'] = [[] for _ in range(self.size)]
            self.buffers['neighbor_pos'] = [[] for _ in range(self.size)]
        # if self.args.alg == 'ours_s':
        #     self.buffers['z'] = np.empty([self.size, self.episode_limit, self.args.noise_dim])
        # if self.args.alg == 'ours' or self.args.alg == 'ours_wvdn':
        #     self.buffers['z'] = np.empty(
        #         [self.size, self.episode_limit, self.n_agents, self.args.noise_dim + self.args.id_dim])
        # if self.args.alg == 'ours_wvdn':
        #     self.buffers['z_team'] = [[] for _ in range(self.size)]
        #     self.buffers['z_king'] = [[] for _ in range(self.size)]
        # thread lock
        self.lock = threading.Lock()

        # store the episode

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']
            if 'ours' in self.args.alg:
                self.buffers['neighbor'][idxs] = episode_batch['neighbor']
                self.buffers['neighbor_pos'][idxs] = episode_batch['neighbor_pos']
            # if 'ours' in self.args.alg:
            #     self.buffers['z'][idxs] = episode_batch['z']
            # if self.args.alg == 'ours_wvdn':
            #     self.buffers['z_team'][idxs] = episode_batch['z_team']
            #     self.buffers['z_king'][idxs] = episode_batch['z_king']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        # if self.args.alg == 'ours_wvdn':
        #     temp_buffer['z_team'] = []
        #     temp_buffer['z_king'] = []
        if 'ours' in self.args.alg:
            temp_buffer['neighbor'] = []
            temp_buffer['neighbor_pos'] = []
        for key in self.buffers.keys():
            if 'neighbor' in key:
                for i in idx:
                    temp_buffer[key].append(self.buffers[key][i])
            else:
                temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
