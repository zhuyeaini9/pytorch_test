import gym
import torch
import numpy as np

e = gym.make('MountainCar-v0')

while True:
    e.reset()
    step = 0
    reward = 0
    while True:
        e.render()
        s,r,done,_ = e.step(1)
        reward += r
        step+=1

        if done:
            print('done',step,'reward',reward)
            break;
