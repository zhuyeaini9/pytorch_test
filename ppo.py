import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import gym
import math
import random
from torch.distributions import Categorical
from torch.distributions import Normal
from collections import namedtuple, deque


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def can_learn(self):
        if len(self.memory) >= self.memory.maxlen:
            return True
        return False

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


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
        self.env_type = self.env_helper.get_env_type()
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
        if self.env_type == 1:
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
            return mu, sigma


class Critic(nn.Module):
    def __init__(self, env, layer_num=2, layer_size=128):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_helper = EnvHelper(env)
        self.model = [
            nn.Linear(self.env_helper.get_obs_space(), layer_size),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(layer_size, layer_size), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(layer_size, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, obs):
        v = self.model(obs)


class PPOAgent(object):
    def __init__(self, env):
        self.env_helper = EnvHelper(env)
        self.env_type = self.env_helper.get_env_type()
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.env = env
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.memory = Replay_Buffer(64 * 64, 64, random.seed(self.seed))
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001)

    def run(self, epsoid=200):
        for i in range(epsoid):
            step(i)

    def pick_action(self, obs):
        if self.env_type == 1:
            prop_discreate = self.actor(torch.tensor(obs).float().unsqueeze(dim=0))
            action = Categorical(prop_discreate)
            return action.sample().item()
        else:
            mu, sigma = self.actor(torch.tensor(obs).float().unsqueeze(dim=0))
            action_distribution = Normal(mu.squeeze(0), sigma.squeeze(0))
            return action_distribution.sample().item()

    def step(self, epsoid_index):
        self.env.seed(self.seed)
        obs = self.env.reset()
        while True:
            action = self.pick_action(obs)
            new_obs, reward, done, _ = self.env.step(action)
            self.memory.add_experience(obs, action, reward, new_obs, done)
            obs = new_obs
            if self.memory.can_learn():
                break
