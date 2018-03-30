#CARTPOLE - Trial1
import gym
import numpy as np

env = gym.make('CartPole-v0')
param = np.random.rand(4)*2 - 1

def runepisode(env,param):
    observation = env.reset()
    total_reward = 0
    
    for i in range(200):
        
        param = np.random.rand(4)*2 - 1
        action = 0 if np.matmul(param,observation)<0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += 1
        
        if done:
            break
    return(total_reward)
    
print(runepisode(env,param))
