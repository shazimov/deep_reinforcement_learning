import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, c1_filters=32, c2_filters=64, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.c1 = nn.Conv2d(in_channels=4, out_channels=c1_filters, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(c1_filters, c2_filters, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(c2_filters, c2_filters, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(c2_filters * 7 * 7, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size, seed, c1_filters=32, c2_filters=64, fc1_1_units=256, fc1_2_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.c1 = nn.Conv2d(4, c1_filters, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(c1_filters, c2_filters, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(c2_filters, c2_filters, kernel_size=3, stride=1)
        self.fc1_1 = nn.Linear(c2_filters * 7 * 7, fc1_1_units)
        self.fc1_2 = nn.Linear(action_size, fc1_2_units)
        self.fc2 = nn.Linear(fc1_1_units+fc1_2_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x1 = F.relu(self.fc1_1(x))
        x2 = F.relu(self.fc1_2(action))
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
