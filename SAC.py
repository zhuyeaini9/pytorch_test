import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from collections import namedtuple, deque
import random
import numpy as np
import gym
import copy
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


class SACNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims):
        super(SACNetwork, self).__init__()
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


class SACAgent(object):
    def __init__(self, env_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.m_env = gym.make(env_name)
        self.action_type = self.get_action_type()
        self.action_size = self.get_action_size()
        self.obs_size = self.get_observation_size()
        assert self.action_type == 2, "Action types must be continuous. Use SAC Discrete instead for discrete actions"

        self.critic_local = SACNetwork(self.obs_size + self.action_size, 20, 20, 1)
        self.critic_local_2 = SACNetwork(self.obs_size + self.action_size, 20, 20, 1)

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=0.02, eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=0.02, eps=1e-4)

        self.critic_target = SACNetwork(self.obs_size + self.action_size, 20, 20, 1)
        self.critic_target_2 = SACNetwork(self.obs_size + self.action_size, 20, 20, 1)

        self.copy_model_over(self.critic_local, self.critic_target)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.memory = Replay_Buffer(1000000, 256, self.seed)
        self.actor_local = SACNetwork(self.obs_size, 20, 20, self.action_size * 2)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=0.003, eps=1e-4)

        self.target_entropy = -torch.prod(torch.Tensor(self.m_env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.003, eps=1e-4)

        self.noise = OU_Noise(self.action_size, self.seed, 0.0, 0.15, 0.25)

    def run_n_episodes(self, num_episodes=450):
        self.episode_number = 0
        self.global_step_number = 0
        self.EPSILON = 1e-6
        while self.episode_number < num_episodes:
            self.reset()
            self.step()

    def step(self):
        eval_ep = self.episode_number % 10 == 0
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(10):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.m_env._max_episode_steps else self.done
            if not eval_ep:
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

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

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1, 5)
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2, 5)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss, 5)
        self.soft_update_of_target_network(self.critic_local, self.critic_target, 0.005)
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, 0.005)
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * 0.99 * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > 1000 and \
               self.enough_experiences_to_learn_from() and self.global_step_number % 20 == 0

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > 256

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.m_env.step(action)
        self.m_env.render()
        self.total_episode_score_so_far += self.reward

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < 1000:
            action = self.m_env.action_space.sample()
            print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)

        action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        nor_ = normal.Normal(mean, std)
        x_t = nor_.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = nor_.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + self.EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def reset(self):
        self.m_env.seed(self.seed)
        self.state = self.m_env.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []

        self.noise.reset()

    def copy_model_over(self, from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def get_action_type(self):
        if isinstance(self.m_env.action_space, gym.spaces.Discrete):
            action_type = 1
        else:
            action_type = 2
        return action_type

    def get_observation_size(self):
        return self.m_env.observation_space.shape[0]

    def get_action_size(self):
        if self.action_type == 1:
            return self.m_env.action_space.n
        if self.action_type == 2:
            return self.m_env.action_space.shape[0]


if __name__ == '__main__':
    gym.logger.set_level(50)
    # MountainCarContinuous-v0
    sac = SACAgent('Pendulum-v0')
    sac.run_n_episodes()
