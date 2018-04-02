# CARTPOLE - Trial1
# CARTPOLE - Hill Climbing
# Check all the neighbouring points. If reward obtained is higher then update parameters 

import gym
import numpy as np

env = gym.make('CartPole-v0')
param = np.random.rand(4)*2 - 1

def runepisode(env,param):
    observation = env.reset()
    total_reward = 0
    
    for i in range(200):
        action = 0 if np.matmul(param,observation)<0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += 1
        
        if done:
            break
    return(total_reward)

#Hill Climbing Starts
best_param = None   
best_reward = 0
noise = 0.1 

for i in range(1000):
    param = np.random.rand(4)*2 - 1
    new_param = param + (np.random.rand(4)*2 - 1)*noise
    reward = runepisode(env,param)
    
    if reward>best_reward:
        best_reward = reward
        param = new_param
        
        if reward == 200:
            break
