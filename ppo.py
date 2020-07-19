import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import gym
import math


class EnvHelper(object):
    def __init__(self, env):
        self.m_env = env
        self.m_env_type = self.get_env_type()

    def get_obs_space(self):
        return self.m_env.observation_space.shape[0]

    def get_action_space(self):
        if self.m_env_type == 1:
            return self.m_env.action_space.n
        else:
            return self.m_env.action_space.shape[0]

    def get_env_type(self):
        if isinstance(self.m_env.action_space, spaces.Discrete):
            action_type = 1
        else:
            action_type = 2
        return action_type

    def get_action_bound(self):
        low_max = abs(max(self.m_env.action_space.low.min(), self.m_env.action_space.low.max(), key=abs))
        high_max = abs(max(self.m_env.action_space.high.min(), self.m_env.action_space.high.max(), key=abs))
        bound_max = max(low_max, high_max)
        if bound_max == math.inf:
            return None
        else:
            return bound_max


class Actor(nn.Module):
    def __init__(self, env, layer_num=2, layer_size=128):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_helper = EnvHelper(env)
        self.model = [
            nn.Linear(self.env_helper.get_obs_space(), layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(layer_size, layer_size), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        if self.env_helper.get_env_type() == 1:
            self.action_discreate = nn.Linear(layer_size, self.env_helper.get_action_space())
        else:
            self.action_con_mu = nn.Linear(layer_size, self.env_helper.get_action_space())
            self.action_con_sigma = nn.Linear(layer_size, self.env_helper.get_action_space())

    def forward(self, obs):
        if self.env_helper.get_env_type() == 1:
            prop_discreate = self.model(obs)
            prop_discreate = F.softmax(self.action_discreate(prop_discreate), dim=-1)
            return prop_discreate
        else:
            action_bound = self.env_helper.get_action_bound()
            prop_con = self.model(obs)
            mu = F.tanh(self.action_con_mu(prop_con))
            if action_bound != None:
                mu = mu * action_bound
            sigma = F.softplus(self.action_con_sigma(prop_con))


class Crict(nn.Module):
    def __init__(self):
        pass
