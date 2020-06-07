import torch
import torch.nn.functional as F
import torch.nn as nn
import gym
from gym import spaces
import torch.optim as optim
from torch.distributions import Categorical, normal
import random
import numpy as np


class Net(nn.Module):
    def __init__(self, input_space, output_space):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class PGAgent(object):
    def __init__(self, env_name):
        self.m_env = gym.make(env_name)
        self.m_env._max_episode_steps = 1000
        self.m_action_type = self.get_action_type()
        self.m_net = Net(self.get_input_space(), self.get_output_space())
        self.m_adam = optim.Adam(params=self.m_net.parameters(), lr=0.01)
        # self.m_adam = optim.SGD(self.m_net.parameters(), lr=0.01, momentum=0.9)

        self.m_barch_reward = []
        self.m_batch_state = []
        self.m_batch_action = []

    def get_input_space(self):
        return self.m_env.observation_space.shape[0]

    def get_output_space(self):
        if self.m_action_type == 1:
            return self.m_env.action_space.n
        else:
            return self.m_env.action_space.shape[0] * 2

    def get_action_type(self):
        if isinstance(self.m_env.action_space, spaces.Discrete):
            action_type = 1
        else:
            action_type = 2

        return action_type

    def get_random(self, step_index, step_all):
        if step_index / step_all < 0.3:
            return 0.3
        if step_index / step_all < 0.6:
            return 0.1
        return 0.01

    def get_action(self, police_action, step_index, step_all):
        if self.m_action_type == 1:
            # if random.random() <= self.get_random(step_index, step_all):
            #     action = random.randint(0, self.get_output_space() - 1)
            #     return action
            # print(police_action)
            # action_distribution = Categorical(police_action)
            # return action_distribution.sample().item()
            action_space = np.arange(self.m_env.action_space.n)
            action = np.random.choice(action_space, p=police_action)
            return action

    def calculate_discounted_rewards_normal(self, reward_list):
        re = self.calculate_discounted_rewards(reward_list)
        return self.normalise_rewards(re)

    def calculate_discounted_rewards(self, reward_list):
        cur_re = 0
        re_discounted_reward = []
        for re in reversed(reward_list):
            re = re + 0.99 * cur_re
            cur_re = re
            re_discounted_reward.append(re)
        re_discounted_reward.reverse()
        return re_discounted_reward

    def normalise_rewards(self, rewards):
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return (rewards - mean_reward) / (std_reward + 1e-8)
        # return rewards - mean_reward

    def reset(self):
        self.m_batch_state = []
        self.m_batch_action = []
        self.m_barch_reward = []

    def step(self, step_index, step_all):
        self.reset()
        state = self.m_env.reset()
        batch = 0
        reward_record = []
        reward_list = []
        action_list = []
        state_list = []
        while True:
            out_action = self.m_net(torch.tensor(state).float().unsqueeze(dim=0)).detach().numpy()

            out_action = out_action.squeeze(0)
            tar_action = self.get_action(out_action, step_index, step_all)
            new_state, reward, done, _ = self.m_env.step(tar_action)

            reward_list.append(reward)
            action_list.append(tar_action)
            state_list.append(state)

            state = new_state

            if done:
                batch += 1

                self.m_barch_reward.extend(self.calculate_discounted_rewards_normal(reward_list))
                self.m_batch_action.extend(action_list)
                self.m_batch_state.extend(state_list)

                reward_record.append(sum(reward_list))

                reward_list = []
                action_list = []
                state_list = []
                state = self.m_env.reset()

                if batch == 5:
                    break

        print(step_index, reward_record)

        self.m_adam.zero_grad()

        state_tensor = torch.FloatTensor(self.m_batch_state)
        reward_tensor = torch.FloatTensor(self.m_barch_reward)
        action_tensor = torch.LongTensor(self.m_batch_action)

        # Calculate loss
        logprob = torch.log(
            self.m_net(state_tensor))
        selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.m_adam.step()

        # for j in range(run_steps):
        #     for i in range(1):
        #         oa = self.m_net(torch.tensor(state).float().unsqueeze(dim=0))
        #         oa = oa.squeeze(0)
        #         if self.m_reward_dis_nor[j] > 0:
        #             if oa[self.m_action_list[j]] > 0.99:
        #                 continue
        #         if self.m_reward_dis_nor[j] < 0:
        #             if oa[self.m_action_list[j]] < 0.01:
        #                 continue
        #         loss = -torch.log(oa[self.m_action_list[j]]) * self.m_reward_dis_nor[j]
        #
        #         self.m_adam.zero_grad()
        #         loss.backward()
        #         self.m_adam.step()

    def run_n_step(self, n):
        for i in range(n):
            self.step(i, n)


agent = PGAgent('CartPole-v0')
agent.run_n_step(1000)
