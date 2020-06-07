import numpy as np


def calculate_discounted_rewards_normal(rewards):
    re = calculate_discounted_rewards(rewards)
    return normalise_rewards(re)


def calculate_discounted_rewards(rewards):
    cur_re = 0
    re_discounted_reward = []
    for re in reversed(rewards):
        re = re + 0.99 * cur_re
        cur_re = re
        re_discounted_reward.append(re)
    re_discounted_reward.reverse()
    return re_discounted_reward

def normalise_rewards(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8)


rewards = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

print(calculate_discounted_rewards_normal(rewards))

rewards = [1,1]
print(calculate_discounted_rewards_normal(rewards))

a = 1000
b = 300
print(b/a)