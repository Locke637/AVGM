import numpy as np
import math


def find_neighbor_id(pos, view_field):
    nei_index = {}
    for id, p in enumerate(pos):
        nei_index[id] = []
        for index, nei_p in enumerate(pos):
            if abs(p[0] - nei_p[0]) < view_field and abs(p[1] - nei_p[1]) < view_field and index != id:
                nei_index[id].append(index)
    return nei_index

def find_neighbor_pos(pos, view_field, num_neighbor):
    # num_neighbor = 3
    # view_field = 5
    nei_index = {}
    nei_pos = {}
    for id, p in enumerate(pos):
        nei_index[id] = []
        nei_pos[id] = []
        d_p_all = {}
        temp_pos = {}
        for index, nei_p in enumerate(pos):
            d_x = abs(p[0] - nei_p[0])
            d_y = abs(p[1] - nei_p[1])
            if d_x < view_field and d_y < view_field and index != id:
                d_p = d_x + d_y
                d_p_all[index] = d_p
                temp_pos[index] = [nei_p[0] - p[0], nei_p[1] - p[1]]
        if d_p_all:
            d_p_all = sorted(d_p_all.items(), key=lambda item: item[1])
            count = 0
            for idpos in d_p_all:
                if count < num_neighbor:
                    nei_index[id].append(idpos[0])
                    nei_pos[id].append(temp_pos[idpos[0]])
                    count += 1
                else:
                    break

    return nei_index, nei_pos


def find_max_q(tmp_q_buffer_list, q_tot):
    comapre_q_max = -10000
    for i in range(pow(2, len(tmp_q_buffer_list))):
        index = []
        find_index_i = i
        while find_index_i / 2 != 0:
            index.append(find_index_i % 2)
            find_index_i = int(find_index_i / 2)
        while len(index) < len(tmp_q_buffer_list):
            index.insert(0, 0)
        tmp_q_tot = np.zeros(2)
        print(index)
        for list_id, id in enumerate(index):
            tmp_q_tot += tmp_q_buffer_list[list_id][id]
        tmp_q_tot += q_tot
        if max(tmp_q_tot) > comapre_q_max:
            comapre_q_max = max(tmp_q_tot)
            return_q_tot = tmp_q_tot
    return return_q_tot


if __name__ == '__main__':
    pos = [[0, 1], [5, 5], [2, 2], [3, 3], [3, 4], [3, 5], [11, 11], [20, 20]]
    # pos = [[0, 10], [5, 15], [5, 10], [10, 5], [10, 0], [10, 1], [10, 3], [10, 2]]
    nei_team, nei_king, nei_pos = find_neighbor_pos(pos, 6)
    print(nei_team, nei_king)
    # print(list(nei_king.values()))
    print(nei_pos)
    # q_list = [[[2, 3], [3, 1]], [[3, 1], [4, 1]], [[4, 5], [5, 6]]]
    # q_list = [[[2, 3], [3, 1]], [[3, 1], [4, 1]]]
    # print(find_max_q(q_list, [0, 0]))
    # index = find_pos_index(pos[0])
    # print(index)
