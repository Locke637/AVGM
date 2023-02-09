import torch.nn as nn
import torch


class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        # print(q_values.shape)
        # return torch.sum(q_values, dim=2, keepdim=True)
        return torch.mean(q_values, dim=2, keepdim=True)


class ShapleyVDNNet(nn.Module):
    def __init__(self):
        super(ShapleyVDNNet, self).__init__()

    def forward(self, q_values):
        ret = []
        for i in range(len(q_values)):
            trans = []
            for j in range(len(q_values[i])):
                # tmp = torch.sum(q_values[i][j], dim=0, keepdim=True)/(q_values[i][j].shape[0])
                tmp = torch.sum(q_values[i][j], dim=0, keepdim=True)
                # print(q_values[i][j], q_values[i][j].shape ,tmp)
                trans.append(tmp)
            trans = torch.stack(trans, dim=0)
            ret.append(trans)
        ret = torch.stack(ret, dim=0)
        ret = ret.cuda()
        # print(ret.shape)
        return ret


class WeightedVDNNet(nn.Module):
    def __init__(self):
        super(WeightedVDNNet, self).__init__()

    def forward(self, q_values, weights):
        # print(q_values)
        # print(weights)
        r = q_values * weights
        # print(r)
        ret = torch.sum(r, dim=2, keepdim=True)
        # print(ret)
        return ret
