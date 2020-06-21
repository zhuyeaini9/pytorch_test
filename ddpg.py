import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
import gym
import copy


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


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class DDPGNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims):
        super(DDPGNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(fc2_dims, output_dims)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DDPGAgent(object):
    def __init__(self, env_name):
        self.global_step_number = 0
        self.game_full_episode_scores = []
        self.rolling_results = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.m_env = gym.make(env_name)
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.m_action_type = self.get_action_type()
        self.critic_local = DDPGNetwork(self.get_observation_size() + self.get_action_size()
                                        , 20, 20, 1)
        self.critic_target = DDPGNetwork(self.get_observation_size() + self.get_action_size()
                                         , 20, 20, 1)
        self.copy_model_params(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters()
                                           , lr=0.02, eps=1e-4)
        self.memory = Replay_Buffer(1000000, 256, random.seed(self.seed))
        self.actor_local = DDPGNetwork(self.get_observation_size()
                                       , 20, 20, self.get_action_size())
        self.actor_target = DDPGNetwork(self.get_observation_size()
                                        , 20, 20, self.get_action_size())
        self.copy_model_params(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters()
                                          , lr=0.003, eps=1e-4)
        self.noise = OU_Noise(self.get_action_size(), self.seed, 0, 0.15, 0.25)

    def copy_model_params(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def get_observation_size(self):
        return self.m_env.observation_space.shape[0]

    def get_action_size(self):
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

    def run_n_episodes(self):
        self.episode_number = 0
        while self.episode_number < 400:
            self.reset()
            self.step()
            self.save_result()
            print(self.episode_number, 'reward:', self.total_episode_score_so_far, 'average-100:',
                  self.rolling_results[-1])

    def reset(self):
        self.m_env.seed(self.seed)
        self.state = self.m_env.reset()
        self.noise.reset()
        self.done = False
        self.total_episode_score_so_far = 0

    def step(self):
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.can_learn():
                for _ in range(10):
                    states, actions, rewards, next_states, dones = self.memory.sample()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-100:]))

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        try:
            return self.m_env.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.m_env.spec.reward_threshold
            except AttributeError:
                return self.m_env.unwrapped.spec.reward_threshold
        return float("inf")

    def update_learning_rate(self, starting_lr, optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done:
            self.update_learning_rate(0.003, self.actor_optimizer)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()

        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss, 5)
        self.soft_update_of_target_network(self.actor_local, self.actor_target, 0.005)

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, 5)
        self.soft_update_of_target_network(self.critic_local, self.critic_target, 0.005)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        loss = F.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = rewards + (0.99 * critic_targets_next * (1.0 - dones))
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def can_learn(self):
        if len(self.memory) > 256 and self.global_step_number % 20 == 0:
            return True
        return False

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None:
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        action = action + self.noise.sample()
        return action.squeeze(0)

    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.m_env.step(action)
        self.m_env.render()
        self.total_episode_score_so_far += self.reward


if __name__ == '__main__':
    gym.logger.set_level(50)
    # ddpg = DDPGAgent('MountainCarContinuous-v0')
    ddpg = DDPGAgent('Pendulum-v0')
    ddpg.run_n_episodes()
