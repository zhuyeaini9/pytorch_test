import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import gym
from gym import spaces
import math
import random
from torch.distributions import Categorical
from torch.distributions import Normal
from collections import namedtuple, deque


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, agent, batch_size=64):
        self.agent = agent
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []
        self.log_prob_action = []

    def can_learn(self, s):
        if len(self.reward) >= s:
            return True
        return False

    def reset(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.next_state.clear()
        self.done.clear()
        self.log_prob_action.clear()

    def add_experience(self, states, actions, rewards, next_states, dones, log_prob_action):
        self.state.append(states)
        self.action.append(actions)
        self.reward.append(rewards)
        self.next_state.append(next_states)
        self.done.append(dones)
        self.log_prob_action.append(log_prob_action)

    def cal(self):
        gaes = []
        with torch.no_grad():
            self.state_tensor = torch.tensor(self.state).float()
            self.next_state_tensor = torch.tensor(self.next_state).float()
            self.log_prob_action_tensor = torch.tensor(self.log_prob_action).float().unsqueeze(dim=1)
            self.action_tensor = torch.tensor(self.action).float().unsqueeze(dim=1)
            self.value = self.agent.critic(self.state_tensor)
            v_ = self.agent.critic(self.next_state_tensor)
            m = (1. - torch.tensor(self.done).float().unsqueeze(dim=1)) * self.agent.gamma
            delta = torch.tensor(self.reward).float().unsqueeze(dim=1) + v_ * m - self.value
            m *= self.agent.gae_lambda
            gae = 0.
            for j in range(len(self.reward) - 1, -1, -1):
                gae = delta[j] + m[j] * gae
                gaes.insert(0, gae)

            gaes = torch.cat(gaes, dim=0).unsqueeze(dim=1)
            self.returns = self.normal(gaes + self.value).unsqueeze(dim=1)
            self.gaes = self.normal(gaes).unsqueeze(dim=1)

    def batchs(self):
        for i in range(0, len(self.done), self.batch_size):
            yield {"state": self.state_tensor[i:i + self.batch_size]
                , "action": self.action_tensor[i:i + self.batch_size]
                , "value": self.value[i:i + self.batch_size]
                , "log_prob_action": self.log_prob_action_tensor[i:i + self.batch_size]
                , "returns": self.returns[i:i + self.batch_size]
                , "gaes": self.gaes[i:i + self.batch_size]}

    def normal(self, rs):
        rs_ = rs.numpy()
        mean_reward = np.mean(rs_)
        std_reward = np.std(rs_)
        return (rs - mean_reward) / (std_reward + 1e-8)


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
        self.device = torch.device('cpu')
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
            return Categorical(prop_discreate)
        else:
            action_bound = self.env_helper.get_action_bound()
            prop_con = self.model(obs)
            mu = F.tanh(self.action_con_mu(prop_con))
            if action_bound != None:
                mu = mu * action_bound
            sigma = F.softplus(self.action_con_sigma(prop_con))
            return Normal(mu, sigma)


class Critic(nn.Module):
    def __init__(self, env, layer_num=1, layer_size=128):
        super().__init__()
        self.device = torch.device('cpu')
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
        return v


class PPOAgent(object):
    def __init__(self, env, learn_step_per_epsoid=1, learn_buffer_size=64 * 10, epsoid=20000, gamma=0.99,
                 gae_lambda=0.95,
                 w_c_loss=.5, w_e_loss=.0):
        self.learn_buffer_size = learn_buffer_size
        self.w_c_loss = w_c_loss
        self.w_e_loss = w_e_loss
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learn_step_per_epsoid = learn_step_per_epsoid
        self.epsoid = epsoid
        self.env_helper = EnvHelper(env)
        self.env_type = self.env_helper.get_env_type()
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.env = env
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.memory = Replay_Buffer(self)
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.0001)

    def run(self):
        for i in range(self.epsoid):
            self.step(i)

    def pick_action(self, obs):
        action_dist = self.actor(torch.tensor(obs).float().unsqueeze(dim=0))
        if self.env_helper.get_env_type() == 1:
            act = action_dist.sample().item()
            return act, action_dist.log_prob(torch.tensor(act).float()).item()
        else:
            act = action_dist.sample()
            return act, action_dist.log_prob(act).item()

    def step(self, epsoid_index):
        obs = self.env.reset()
        self.memory.reset()

        while True:
            action, log_prop_action = self.pick_action(obs)
            new_obs, reward, done, _ = self.env.step(action)
            self.memory.add_experience(obs, action, reward, new_obs, done, log_prop_action)
            if done:
                new_obs = self.env.reset()

            obs = new_obs
            if self.memory.can_learn(self.learn_buffer_size):
                break

        self.learn()
        print(epsoid_index, self.test())

    def test(self):
        reward_test_list = []
        obs = self.env.reset()
        while True:
            action_test, _ = self.pick_action(obs)
            new_obs_test, reward_test, done_test, _ = self.env.step(action_test)
            reward_test_list.append(reward_test)
            if done_test:
                return sum(reward_test_list)
            obs = new_obs_test

    def learn(self):
        self.memory.cal()
        for _ in range(self.learn_step_per_epsoid):
            for b in self.memory.batchs():
                dist = self.actor(b["state"])
                value = self.critic(b["state"])

                actor_loss = (-dist.log_prob(b["action"].squeeze(dim=1)) * b["gaes"]).mean()
                critic_loss = (b["returns"] - value).pow(2).mean()

                e_loss = dist.entropy().mean()

                loss = actor_loss + self.w_c_loss * critic_loss - self.w_e_loss * e_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


class Batch(object):
    def __init__(self, agent, states, actions, rewards, next_states, dones, log_prob_action):
        self.agent = agent
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.log_prob_action = log_prob_action
        self.cal()

    def cal(self):
        gaes = []
        with torch.no_grad():
            self.value = self.agent.critic(self.states)
            v_ = self.agent.critic(self.next_states)
            m = (1. - self.dones) * self.agent.gamma
            delta = self.rewards + v_ * m - self.value
            m *= self.agent.gae_lambda
            gae = 0.
            for j in range(len(self.rewards) - 1, -1, -1):
                gae = delta[j] + m[j] * gae
                gaes.insert(0, gae)

            gaes = torch.cat(gaes, dim=0).unsqueeze(dim=1)
            self.returns = self.normal(gaes + v).unsqueeze(dim=1)
            self.gaes = self.normal(gaes).unsqueeze(dim=1)

    def normal(self, rs):
        rs_ = rs.numpy()
        mean_reward = np.mean(rs_)
        std_reward = np.std(rs_)
        return (rs - mean_reward) / (std_reward + 1e-8)


if __name__ == '__main__':
    gym.logger.set_level(50)
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCarContinuous-v0")
    agent = PPOAgent(env)
    agent.run()
