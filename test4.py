import gym
import torch
import numpy as np

e = gym.make('MountainCarContinuous-v0')

while True:
    e.reset()
    step = 0
    while True:
        e.render()
        s,r,done,_ = e.step(torch.tensor([10]))

        step+=1

        if done:
            print('done',step)
            break;
