
# coding: utf-8

# In[3]:


import numpy as np
import gym


# In[4]:


env = gym.make('CartPole-v0')


# In[5]:


param = np.random.rand(4)     


# In[6]:


def policy(p,o):
    ans = np.dot(o,p)
    if ans>=0:
        return(1)
    else:
        return(0)


# In[7]:


# Delta Theta
high = env.observation_space.high
low = env.observation_space.low

Dt = np.array([(high[0]-low[0])/100,0.01,(high[2]-low[2])/100,0.01])


# In[31]:


episodes = 100
Jtest = 0
for ep in range(episodes):
    obs = env.reset()
    for t in range(200):
        #env.render()
        action = policy(param,obs)
        obs, r, done, info = env.step(action)
        Jtest += r
        if done:
            break
print(Jtest/100)


# In[32]:


# We will be using finite differences method for Performance evaluation
# This algorithm uses forward difference estimators

for z in range(200):
    # J reference
    J = 0
    episodes = 1
    for ep in range(episodes):
        obs = env.reset()

        for t in range(200):
            #env.render()
            action = policy(param,obs)
            obs, r, done, info = env.step(action)
            J += r

            if done:
                break

    # J delta
    Jd = np.array([0 for x in range(4)])
    for i in range(4):
        param_new = np.copy(param)
        param_new[i] += Dt[i]

        episodes = 1
        for ep in range(episodes):
            obs = env.reset()

            for t in range(200):
                #env.render()
                action = policy(param_new,obs)
                obs, r, done, info = env.step(action)
                Jd[i] += r

                if done:
                    break

    # This algorithm uses linear parameterization
    lr = 0.01
    for i in range(4):
        param[i] = param[i] + lr*(Jd[i]-J)


# In[33]:


episodes = 1
Jtest = 0
for ep in range(episodes):
    obs = env.reset()
    for t in range(200):
        #env.render()
        action = policy(param,obs)
        obs, r, done, info = env.step(action)
        Jtest += r
        if done:
            break
print(Jtest/100)

