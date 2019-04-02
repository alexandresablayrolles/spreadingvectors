import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net import Normalize, forward_pass, StraightThroughQuantizer

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, post_act=False):
        super(Residual, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=not batch_norm)

        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(out_channels)
        self.post_act = post_act

        if in_channels != out_channels:
            layers = [nn.Linear(in_channels, out_channels, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.residual = nn.Sequential(*layers)
        else:
            self.residual = nn.Sequential()


    def forward(self, x):
        y = x
        y = self.linear(y)
        y = self.bn(y)
        if self.post_act:
            y = F.relu(y, inplace=False)

        x = y + self.residual(x)

        return x


def createNet(arch, residual=False, batch_norm=False, post_act=True):
    layer_sz = [int(x) for x in arch.split("-")]
    layers = []
    for i_layer in range(len(layer_sz) - 1):

        if residual:
            layers.append(Residual(layer_sz[i_layer], layer_sz[i_layer + 1], batch_norm=batch_norm, post_act=post_act))
            if not post_act and i_layer < len(layer_sz) - 2:
                layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(layer_sz[i_layer], layer_sz[i_layer+1], bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sz[i_layer+1]))
            if i_layer < len(layer_sz) - 2:
                layers.append(nn.ReLU())

    layers.append(Normalize())

    return nn.Sequential(*layers)
    # net = nn.Sequential(
    #     nn.Linear(in_features=dim, out_features=dint, bias=True),
    #     nn.BatchNorm1d(dint),
    #     nn.ReLU(),
    #     nn.Linear(in_features=dint, out_features=dint, bias=True),
    #     nn.BatchNorm1d(dint),
    #     nn.ReLU(),
    #     nn.Linear(in_features=dint, out_features=dout, bias=True),
    #     Normalize()
    # )
