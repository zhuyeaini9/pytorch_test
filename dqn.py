from collections import Counter
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import gym
from collections import namedtuple, deque


class Epsilon_Greedy_Exploration(object):
    """Implements an epsilon greedy exploration strategy"""

    def __init__(self):
        self.notified_that_exploration_turned_off = False
        self.exploration_cycle_episodes_length = None
        self.random_episodes_to_run = 0

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)

        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = 1

        if self.exploration_cycle_episodes_length is None:
            epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        else:
            epsilon = self.calculate_epsilon_with_cyclical_strategy(episode_number)
        return epsilon

    def calculate_epsilon_with_cyclical_strategy(self, episode_number):
        """Calculates epsilon according to a cyclical strategy"""
        max_epsilon = 0.5
        min_epsilon = 0.001
        increment = (max_epsilon - min_epsilon) / float(self.exploration_cycle_episodes_length / 2)
        cycle = [ix for ix in range(int(self.exploration_cycle_episodes_length / 2))] + [ix for ix in range(
            int(self.exploration_cycle_episodes_length / 2), 0, -1)]
        cycle_ix = episode_number % self.exploration_cycle_episodes_length
        epsilon = max_epsilon - cycle[cycle_ix] * increment
        return epsilon

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass


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


class DQNNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(object):
    def __init__(self, env_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = random.randint(0, 2 ** 32 - 2)
        self.m_env = gym.make(env_name)
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.action_type = self.get_action_type()
        self.action_size = self.get_action_size()
        self.obs_size = self.get_observation_size()

        self.memory = Replay_Buffer(40000, 256, self.seed)
        self.q_network_local = DQNNetwork(self.obs_size, 30, 15, self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=0.01, eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration()

    def get_score_required_to_win(self):
        try:
            return self.m_env.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.m_env.spec.reward_threshold
            except AttributeError:
                return self.m_env.unwrapped.spec.reward_threshold

    def run_n_episodes(self, num_episodes=450):
        self.episode_number = 0
        self.rolling_results = []
        self.turn_off_exploration = False
        self.global_step_number = 0
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            print(self.episode_number,self.total_episode_score_so_far)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(1):
                    self.learn()
            self.save_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences()
        else:
            states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, 0.7)

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (0.99 * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        # must convert actions to long so can be used as index
        Q_expected = self.q_network_local(states).gather(1, actions.long())
        return Q_expected

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

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None:
            state = self.state
        if isinstance(state, np.int64) or isinstance(state, int):
            state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        self.q_network_local.eval()  # puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()  # puts network back in training mode
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.m_env.step(action)
        self.m_env.render()
        self.total_episode_score_so_far += self.reward

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.global_step_number % 1 == 0 and len(self.memory) > 256

    def reset_game(self):
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
        self.exploration_strategy.reset()

        self.update_learning_rate(0.01, self.q_network_optimizer)

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
    dqn = DQNAgent('MountainCar-v0')
    dqn.run_n_episodes()
