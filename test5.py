import torch
from torch.distributions import Categorical,normal
import numpy as np
import math

n = normal.Normal(0,1)
s = n.sample()
l = n.log_prob(s)
for i in np.arange(-2.0,2.0,0.1):
    print(i,n.log_prob(i),math.pow(math.e,n.log_prob(i)))