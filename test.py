import gym
import gym.spaces
import numpy as np
import math

env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
ob = env.observation_space



h = ob.high
l = ob.low




kkk = gym.spaces.Box(low=np.array([-1.0, math.inf]), high=np.array([2.0, 4.0]), dtype=np.float32)
n = abs(max(kkk.low.min(), kkk.low.max(), key=abs))
if n == math.inf:
    print('nnn')



#print(abs(max(kkk.low.all(), key=abs)))

