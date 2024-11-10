#!/usr/bin/env python
# coding: utf-8

# ## Large Policy v3: Q-Learning
# 
# Given unknown `(s, a, r, sp)` data, find optimal policy. Not all `(s, a)` pairs will be seen in data, so interpolate from neighbors.
# - States: |S| = 302020
# - Actions: 9 actions
# - Discount factor = 0.95
# 
# Lisa Fung
# 
# Last Updated: 11/9/24

# ### Data Exploration

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats


# In[ ]:


large_data = pd.read_csv("../data/large.csv")


# In[3]:


n_states = 302020
n_actions = 9


# ### Approach
# 

# Use Q-learning to find action-value function $Q(s, a)$.
# - Run through data multiple times to update $Q(s, a) \leftarrow Q(s, a) + \alpha (r + \gamma \max_a Q(s', a') - Q(s, a))$

# ### Q-learning

# In[11]:


# Timing function
import time
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        return result, elapsed_time
    return wrapper


# In[8]:


@time_it
def q_learn(Q, data, iters=1, learning_rate=0.1, discount_rate=0.95):
    for it in range(iters):
        for i in range(len(data)):
            s, a, r, sp = data.iloc[i]
            Q[s, a] += learning_rate * (r + discount_rate * max(Q[sp, :]) - Q[s, a])
    return Q


# In[120]:


def grid_display(data, xdim, ydim, fig_title=None):
    # data = data[1:, 1:].reshape((xdim, ydim, zdim))

    fig, axes = plt.subplots(1, figsize=(20, 15), sharex=True, sharey=True)

    # # Maximum, minimum value range for colorbar
    # vmin, vmax = data.min(), data.max()

    # for i in range(7):
    #     img = axes[i].imshow(data[:, :, i], cmap='viridis', vmin=vmin, vmax=vmax)
    #     axes[i].set_title(f"Action {i+1}")

    

    # plt.tight_layout()
    
    img = axes.matshow(data[1:, 1:].reshape(xdim, ydim))#, cmap='viridis', vmin=0, vmax=1)
    # plt.colorbar()
    # plt.colorbar(img, ax=axes, fraction=0.046, pad=0.04, shrink=0.8)
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)

    if fig_title is not None:
        plt.savefig(f"{fig_title}.png", dpi=300)


# In[9]:


Q = np.zeros((n_states + 1, n_actions + 1))
q_learn(Q, large_data)


# In[34]:


# Store results: (iterations, learning_rate) : (Q, runtime)
# Q_results = {}

for n_iter in [30]:
    for lr in [0.01, 0.05, 0.1, 0.2]:
        Q = np.zeros((n_states + 1, n_actions + 1))
        Q_results[(n_iter, lr)] = q_learn(Q, large_data, iters=n_iter, learning_rate=lr)


# In[35]:


# Q_results


# In[ ]:


# np.save("large_Q_function_sarsa_100iters.npy", Q)
# loaded_arr = np.load("large_Q_function_sarsa_100iters.npy")


# ### Extract Policy from Q Function

# In[32]:


# Extract optimal policy pi(s) = a from action-value function Q(s, a)

def extract_policy(Q, mode='random'):
    """
    """
    policy = np.zeros(n_states+1)
    # predicable_action = np.random.randint(1, n_actions+1)
    # predicable_action = 4
    for s in range(1, n_states+1):
        policy[s] = np.argmax(Q[s, 1:])+1
        # if policy[s] not in [1, 2, 3, 4]: # Actions [5,9] are usually 0, random
        #     if mode == 'random':
        #         policy[s] = np.random.randint(1, 5)
        #     if mode == 'previous':
        #         policy[s] = predicable_action
        # else:
        #     predicable_action = policy[s]

    return policy


# In[31]:


# np.unique(optimal_policy_qlearn, return_counts=True)
np.argmax([1, 2, 3][1:])


# In[25]:


# Write optimal policy to file
def write_policy(policy, policy_name):
    with open(f"{policy_name}.policy", "w") as file:
        for a in policy[1:]:
            file.write(f"{int(a)}\n")


# In[36]:


# optimal_policy_qlearn = extract_policy(Q, mode='random')

for (n_iters, lr), (Q, runtime) in Q_results.items():
    if n_iters == 30:
        policy = extract_policy(Q, mode='random')
        print(f"Policy from n_iters={n_iters}, lr={lr}:")
        print(np.unique(policy, return_counts=True))
        print()
        write_policy(policy, f"large_qlearn_iters{n_iters}_lr{lr}")


# In[41]:


# Save Q_results
import pickle

with open("large_qlearn_results.pkl", "wb") as f:
    pickle.dump(Q_results, f)


# In[49]:


with open("large_qlearn_results.pkl", "rb") as f:
    loaded_results = pickle.load(f)


# In[ ]:




