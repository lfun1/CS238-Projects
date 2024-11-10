#!/usr/bin/env python
# coding: utf-8

# ## Medium Policy v1
# 
# Given Mountain Car problem, `(s, a, r, sp)` data, find optimal policy. Not all `(s, a)` pairs will be seen in data, so interpolate from neighbors.
# - States: |S| = 50,000 for 500 position values, 100 velocity values. `s = 1 + pos + 500*vel`
# - Actions: 7 actions with different amounts of acceleration
#     - Could try to decode based on position/velocity change
# - Reward: R(s, a)
# 
# Lisa Fung
# 
# Last Updated: 11/9/24

# ### Data Exploration

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice


# In[7]:


medium_data = pd.read_csv("../data/medium.csv")


# In[8]:


medium_data.head()
print("Unique rewards:", sorted(medium_data['r'].unique()))
print(f"Data contains {len(medium_data['s'].unique())} unique states s")
print(f"Data contains {len(medium_data['sp'].unique())} unique states sp")
print()
print("Action, reward pairs:")
print(medium_data[['a', 'r']].value_counts().sort_index(level=[0, 1]))
print()
print("Data of finishing states (reached flag):")
print(medium_data[medium_data['r'] > 100000 - 250].hist())


# Medium Data Observations
# 
# Rewards
# - Rewards depend only on actions, and whether flag is reached at state.
# - $R(s = flag) = +100000$
# - $R(a) = -c \cdot |a - 4|$
#     - Accelerate most: $a = 1, 7$ with penalty -225. 
#     - No acceleration: $a = 4$ with penalty 0.
# 

# ### Approach
# 

# Use Q-learning or SARSA (whichever simpler to implement) to find action-value function $Q(s, a)$.
# - Run through data multiple times to update $Q(s, a) \leftarrow Q(s, a) + \alpha (r + \gamma Q(s', a') - Q(s, a))$

# ### SARSA

# In[9]:


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
        return result#, elapsed_time
    return wrapper


# In[10]:


n_pos = 500
n_vel = 100
n_actions = 7
Q = np.zeros((n_pos * n_vel + 1, n_actions + 1))


# In[ ]:


@time_it
def sarsa(Q, data, iters=50, learning_rate=0.1, discount_rate=1):
    for it in range(iters):
        for i in range(len(data)-1):
            s, a, r, sp = data.iloc[i]
            sp, ap, rp, spp = data.iloc[i+1]
            Q[s, a] += learning_rate * (r + discount_rate * Q[sp, ap] - Q[s, a])
    return Q


# In[12]:


def grid_display(data, xdim, ydim, zdim, fig_title=None):
    data = data[1:, 1:].reshape((xdim, ydim, zdim))

    fig, axes = plt.subplots(zdim, figsize=(20, 15), sharex=True, sharey=True)

    # Maximum, minimum value range for colorbar
    vmin, vmax = data.min(), data.max()

    for i in range(7):
        img = axes[i].imshow(data[:, :, i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Action {i+1}")

    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)

    # plt.tight_layout()
    
    # for z in range(1, zdim+1):
    #     print(f"Action a={z}")
    #     img = axes[z-1].matshow(x[1:,z].reshape(xdim, ydim), cmap='viridis', vmin=0, vmax=1)
    #     # plt.colorbar()
    #     plt.colorbar(img, ax=axes[z-1], fraction=0.046, pad=0.04, shrink=0.8)
    if fig_title is not None:
        plt.savefig(f"{fig_title}.png", dpi=300)


# In[14]:


Q = sarsa(Q, medium_data, iters=50, learning_rate=0.1)
print(Q)


# In[15]:


grid_display(Q, n_vel, n_pos, n_actions, "medium_Q_sarsa_iters50_lr0.1")


# In[16]:


np.save("Q_function_sarsa_50iters_lr0.1.npy", Q)
loaded_arr = np.load("Q_function_sarsa_50iters_lr0.1.npy")


# ### Extract Policy from Q Function

# In[17]:


# Extract optimal policy pi(s) = a from action-value function Q(s, a)
n_states = n_pos * n_vel

@time_it
def extract_policy(Q, mode='random'):
    policy = np.zeros(n_states+1)
    # nonzero_action = np.random.randint(1, n_actions+1)
    nonzero_action = 7  # start with big acceleration to the right?
    for s in range(1, n_states+1):
        policy[s] = np.argmax(Q[s, :])
        if policy[s] == 0:  # State not visited
            if mode == 'random':
                policy[s] = np.random.randint(1, n_actions+1)
            if mode == 'previous':
                policy[s] = nonzero_action
        else:
            nonzero_action = policy[s]


    return policy


# In[18]:


optimal_policy_sarsa = extract_policy(Q, mode='previous')


# In[19]:


np.unique(optimal_policy_sarsa, return_counts=True)


# In[20]:


with open("medium.policy", "w") as file:
    for a in optimal_policy_sarsa[1:]:
        file.write(f"{int(a)}\n")


# In[21]:


medium_time = 193.735866 + 0.108739
medium_time


# In[ ]:





# ### Future Improvements
# 
# - Average values of Q(s, a) with some distance-dependent discount for unvisited states s

# In[ ]:




