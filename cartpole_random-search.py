#CARTPOLE - Trial1
#CARTPOLE - Random Search

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

#Random Search Starts
best_param = None   
best_reward = 0
avg_rwd = []

for j in range(1000):
    for i in range(1000):
        param = np.random.rand(4)*2 - 1
        reward = runepisode(env,param)
        if reward>best_reward:
            best_reward = reward
            best_param = param
            
            if reward == 200:
                avg_rwd.append(i)
                break
