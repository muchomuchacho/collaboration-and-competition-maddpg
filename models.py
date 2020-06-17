import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size,
                 hidden_in_dim=200, hidden_out_dim=150):
        super(Actor, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(state_size, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, action_size)
        self.nonlin = F.relu  # leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        # return a vector of the force
        h1 = self.nonlin(self.fc1(x))

        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return torch.tanh(h3)


class Critic(nn.Module):
    def __init__(self, state_size, action_size,
                 hidden_in_dim=200, hidden_out_dim=150):
        super(Critic, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(state_size*2, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim+(action_size*2), hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, 1)
        self.nonlin = F.relu  # leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x, a):
        # critic network simply outputs a number
        s = torch.cat((x, a), dim=1)
        h1 = self.nonlin(self.fc1(s))
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return h3
