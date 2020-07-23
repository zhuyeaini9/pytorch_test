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

    def __init__(self, learn_size=64 * 64, batch_size=64):
        self.memory = []
        self.learn_size = learn_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "log_prob_action"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def reset(self):
        self.memory.clear()

    def can_learn(self):
        if len(self.memory) >= self.learn_size:
            return True
        return False

    def add_experience(self, states, actions, rewards, next_states, dones, log_prob_action):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done, log_prob_action)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones, log_prob_action)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones, log_prob_action)
            self.memory.append(experience)

    def batchs(self, batch_size=None):
        if batch_size is not None:
            bs = batch_size
        else:
            bs = self.batch_size

        for i in range(0, len(self.memory), bs):
            experiences = self.memory[i:i + bs]
            states, actions, rewards, next_states, dones, log_prob_action = self.separate_out_data_types(experiences)
            yield [states, actions, rewards, next_states, dones, log_prob_action]

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones, log_prob_action = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones, log_prob_action
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
        log_prob_action = torch.from_numpy(
            np.vstack([e.log_prob_action for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones, log_prob_action

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
        return v


class PPOAgent(object):
    def __init__(self, env, learn_step_per_epsoid=2, epsoid=200, gamma=0.9, gae_lambda=0.9, eps_clip=0.3,
                 max_grad_norm=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.learn_step_per_epsoid = learn_step_per_epsoid
        self.epsoid = epsoid
        self.env_helper = EnvHelper(env)
        self.env_type = self.env_helper.get_env_type()
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.env = env
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.memory = Replay_Buffer()
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.001)

    def run(self):
        for i in range(self.epsoid):
            self.step(i)

    def pick_action(self, obs):
        action_dist = self.actor(torch.tensor(obs).float().unsqueeze(dim=0))
        act = action_dist.sample().item()
        return act, action_dist.log_prob(torch.tensor(act).float()).item()

    def step(self, epsoid_index):
        self.env.seed(self.seed)
        obs = self.env.reset()
        self.memory.reset()
        while True:
            action, log_prop_action = self.pick_action(obs)
            new_obs, reward, done, _ = self.env.step(action)
            self.memory.add_experience(obs, action, reward, new_obs, done, log_prop_action)
            obs = new_obs
            if self.memory.can_learn():
                self.learn()
                break

    def learn(self):
        batchs = []
        with torch.no_grad():
            for b in self.memory.batchs():
                batchs.append(Batch(self, b[0], b[1], b[2], b[3], b[4], b[5]))
        for _ in range(self.learn_step_per_epsoid):
            for b in batchs:
                dist = self.actor(b.states)
                value = self.critic(b.states)

                ratio = (dist.log_prob(b.actions.squeeze(dim=1)) - b.log_prob_action.squeeze(dim=1)).exp().float()
                surr1 = ratio * b.gaes
                surr2 = ratio.clamp(1. - self.eps_clip, 1. + self.eps_clip) * b.gaes
                actor_loss = -torch.min(surr1, surr2).mean()

                v_clip = b.value + (value - b.value).clamp(-self.eps_clip, self.eps_clip)
                vf1 = (b.value - value).pow(2)
                vf2 = (b.value - v_clip).pow(2)
                critic_loss = .5 * torch.max(vf1, vf2).mean()

                e_loss = dist.entropy().mean()

                loss = actor_loss + critic_loss + e_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.max_grad_norm)
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
        self.gaes = []
        self.value = []
        self.cal()
        self.value = torch.cat(self.value,dim=0)
        self.gaes = torch.cat(self.gaes,dim=0)

    def cal(self):
        with torch.no_grad():
            v = self.agent.critic(self.states)
            self.value.append(v)
            v_ = self.agent.critic(self.next_states)
            m = (1. - self.dones) * self.agent.gamma
            delta = self.rewards + v_ * m - v
            m *= self.agent.gae_lambda
            gae = 0.
            for j in range(len(self.rewards) - 1, -1, -1):
                gae = delta[j] + m[j] * gae
                self.gaes.insert(0, gae)

            self.normalise_rewards(self.gaes)

    def normalise_rewards(self, rewards):
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return (rewards - mean_reward) / (std_reward + 1e-8)


if __name__ == '__main__':
    gym.logger.set_level(50)
    env = gym.make("CartPole-v0")
    agent = PPOAgent(env)
    agent.run()
