import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv

class Actor(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self._max = max_action

    def forward(self, s, **kwargs):
        s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
        return logits, None


class ActorProb(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape,
                 max_action, device='cpu', unbounded=False):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Linear(128, np.prod(action_shape))
        self._max = max_action
        self._unbounded = unbounded

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        if not self._unbounded:
            mu = self._max * torch.tanh(self.mu(logits))
        else:
            mu = self.mu(logits)
        sigma = torch.exp(self.sigma(logits))
        return (mu, sigma), None


class Critic(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if a is not None and not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is None:
            logits = self.model(s)
        else:
            a = a.view(batch, -1)
            logits = self.model(torch.cat([s, a], dim=1))
        return logits


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='Ant-v2')
    parser.add_argument('--task',type=str,default='MountainCarContinuous-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=bool, default=True)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_sac(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    train_envs = gym.make(args.task)
    # train_envs = VectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = gym.make(args.task)
    # test_envs = SubprocVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor = ActorProb(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device, unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = Critic(
        args.layer_num, args.state_shape, args.action_shape, args.device
    ).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(
        args.layer_num, args.state_shape, args.action_shape, args.device
    ).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma, args.alpha,
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=args.rew_norm, ignore_done=True)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_sac()
