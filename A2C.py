import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import gym
from torch.distributions import normal

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array(
            [np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)
        self.fc4 = nn.Linear(fc2_dims, output_dims)

    def forward(self, observation):
        state = torch.FloatTensor(observation)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.fc3(x))
        var = F.softplus(self.fc4(x))

        return mu,var

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

    def forward(self, observation):
        state = torch.FloatTensor(observation)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Env_Utility(object):
    def __init__(self, env):
        self.m_env = env
        self.m_action_type = self.get_action_type()

    def get_input_space(self):
        return self.m_env.observation_space.shape[0]

    def get_output_space(self):
        if self.m_action_type == 1:
            return self.m_env.action_space.n
        if self.m_action_type == 2:
            return self.m_env.action_space.shape[0]

    def get_action_type(self):
        if isinstance(self.m_env.action_space, gym.spaces.Discrete):
            action_type = 1
        else:
            action_type = 2
        return action_type


class Agent(object):
    def __init__(self, env, noise, ut, gamma=0.99, layer1_size=64, layer2_size=64):
        self.gamma = gamma
        self.env_utility = ut
        self.actor = ActorNetwork(self.env_utility.get_input_space()
                                    , 64, 64
                                    , self.env_utility.get_output_space())
        self.critic = CriticNetwork(self.env_utility.get_input_space() + self.env_utility.get_output_space()
                                     , 64, 64
                                     , 1)
        self.actor_opti = optim.Adam(params=self.actor.parameters(), lr=0.0005)
        self.critic_opti = optim.Adam(params=self.critic.parameters(), lr=0.0005)

        self.noise = noise

    def choose_action(self, observation):

        mu,var = self.actor.forward(observation)

        tar_nor = normal.Normal(mu.squeeze(0),var.squeeze(0))
        tar_val = tar_nor.sample()
        return tar_val.unsqueeze(0),tar_nor.log_prob(tar_val)

    def learn(self, state, action, reward, new_state, done):
        next_action, _ = agent.choose_action(new_state)
        critic_value_next = self.critic.forward(
            torch.cat((torch.FloatTensor(new_state), torch.FloatTensor(next_action.float())), 0)).detach()
        critic_value = self.critic.forward(torch.cat((torch.FloatTensor(state),torch.FloatTensor(action.float())),0))

        reward = torch.tensor(reward).unsqueeze(0)

        ada = reward + self.gamma * critic_value_next * (1 - int(done)) - critic_value
        critic_loss = F.mse_loss(reward + self.gamma * critic_value_next * (1 - int(done)), critic_value)
        self.critic_opti.zero_grad()
        critic_loss.backward()
        self.critic_opti.step()

        act_pre,log_prob = self.choose_action(state)
        actor_loss = -log_prob*ada.item()
        #actor_loss = -self.critic(torch.cat((torch.FloatTensor(state), torch.FloatTensor(act_pre)), 0)).mean()
        self.actor_opti.zero_grad()
        actor_loss.backward()
        self.actor_opti.step()


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    ut = Env_Utility(env)
    noise = OU_Noise(ut.get_output_space(), 2)

    agent = Agent(env=env, ut=ut, noise=noise)
    for i in range(100):
        done = False
        observation = env.reset()
        noise.reset()
        score = 0
        while not done:
            action,_ = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        print('episose', i, 'score:', score)
