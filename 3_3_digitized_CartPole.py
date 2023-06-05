#!/usr/bin/env python
# coding: utf-8

# ## 3.3 CartPoleの状態を離散化してみる

# In[1]:


# 使用するパッケージの宣言
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import gym


# In[2]:


# 定数の設定
ENV = 'CartPole-v0'  # 使用する課題名
NUM_DIZITIZED = 6  # 各状態の離散値への分割数


# In[3]:


# CartPoleを実行してみる
env = gym.make(ENV)  # 実行する課題を設定
observation = env.reset()  # 環境の初期化


# In[4]:


# 離散化の閾値を求める


def bins(clip_min, clip_max, num):
    '''観測した状態（連続値）を離散値にデジタル変換する閾値を求める'''
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


# In[5]:


print(np.linspace(-2.4, 2.4, 6 + 1))


# In[6]:


print(np.linspace(-2.4, 2.4, 6 + 1)[1:-1])


# In[10]:


def digitize_state(observation):
    '''観測したobservation状態を、離散値に変換する'''
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED))]
    return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])


# In[11]:


digitize_state(observation)
plt.show()

# In[ ]:




