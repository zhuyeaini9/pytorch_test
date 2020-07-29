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
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "log_prob_action"])
        self.device = torch.device("cpu")
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []
        self.log_prob_action = []

    def reset(self):
        self.memory.clear()
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.next_state.clear()
        self.done.clear()
        self.log_prob_action.clear()

    def add_experience2(self, states, actions, rewards, next_states, dones, log_prob_action):
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

    def batchs2(self):
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
    def __init__(self, env, learn_step_per_epsoid=1, collect_env_step=10, epsoid=20000, gamma=0.99, gae_lambda=0.95,
                 eps_clip=0.3,
                 max_grad_norm=.5, w_c_loss=.5, w_e_loss=.0):
        self.collect_env_step = collect_env_step
        self.w_c_loss = w_c_loss
        self.w_e_loss = w_e_loss
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
        self.memory = Replay_Buffer(self)
        self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.0001)

    def run(self):
        for i in range(self.epsoid):
            self.step(i)

    def pick_action(self, obs):
        action_dist = self.actor(torch.tensor(obs).float().unsqueeze(dim=0))
        act = action_dist.sample().item()
        return act, action_dist.log_prob(torch.tensor(act).float()).item()

    def step(self, epsoid_index):
        obs = self.env.reset()
        self.memory.reset()

        rewards = []
        rewards_sum = []
        cur_env_step = 1
        while True:
            action, log_prop_action = self.pick_action(obs)
            new_obs, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            self.memory.add_experience2(obs, action, reward, new_obs, done, log_prop_action)
            if done:
                rewards_sum.append(sum(rewards))
                rewards = []
                new_obs = self.env.reset()
                cur_env_step += 1
                if cur_env_step >= self.collect_env_step:
                    ave_reward = np.mean(rewards_sum)
                    if ave_reward >= 190 or epsoid_index % 200 == 0:
                        print("epsoid:", epsoid_index, " reward:", ave_reward, "test:", self.test())

                    rewards_sum = []
                    break
            obs = new_obs

        self.learn()

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
            for b in self.memory.batchs2():
                dist = self.actor(b["state"])
                value = self.critic(b["state"])

                ratio = (dist.log_prob(b["action"].squeeze(dim=1)) - b["log_prob_action"].squeeze(dim=1)).exp().float()
                ratio = ratio.unsqueeze(dim=1)
                surr1 = ratio * b["gaes"]
                surr2 = ratio.clamp(1. - self.eps_clip, 1. + self.eps_clip) * b["gaes"]
                actor_loss = -torch.min(surr1, surr2).mean()

                v_clip = b["value"] + (value - b["value"]).clamp(-self.eps_clip, self.eps_clip)
                vf1 = (b["returns"] - value).pow(2)
                vf2 = (b["returns"] - v_clip).pow(2)
                critic_loss = .5 * torch.max(vf1, vf2).mean()
                # critic_loss = (b.returns - value).pow(2).mean()

                e_loss = dist.entropy().mean()

                loss = actor_loss + self.w_c_loss * critic_loss - self.w_e_loss * e_loss

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
    agent = PPOAgent(env)
    agent.run()
