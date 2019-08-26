import gym
import gym_cassie
import numpy as np
import pickle
import torch
import time
import torchvision
from gym_cassie.envs.cassiemujoco import pd_in_t
from random import randint
from rl.policies import GaussianMLP
import matplotlib.pyplot as plt

env = gym.make("Cassie-mimic-v0")
action_queue = []
policy_step = torch.load('/home/beast/srikar/apex/trained_models/apex/global_policy.pt')

state = env.reset()
print(env.get_full_state('stand').shape)
for i in range(10 ** 10):
    state = torch.Tensor(state)
    # print(model(state[0:40].unsqueeze(0).float()))
    #_, action = policy_step.act(torch.from_numpy(env.get_full_state('stand')).float(), deterministic = True)
    action = policy_step(torch.from_numpy(env.get_full_state('stand')).float())
    # if(len(action_queue) == 0):
    #     action0 = action
    #     for i in range(4):
    #         action_queue.append(action0)
    # # else:
    #     action_queue.insert(0, action)
    state, reward, done, info = env.step(action.data.numpy())

    plt.scatter(i, 0.030*np.exp(-env.height_diff))
    plt.scatter(i, 0.015*np.exp(-env.pel_vel))    
    plt.scatter(i, 0.020*np.exp(-0.01 * env.motor_torque))
    plt.scatter(i, 0.015*np.exp(-env.vel_diff))
    plt.scatter(i, 0.020*np.exp(-env.theta_diff))
    plt.pause(0.05)
    plt.show()
    env.compute_reward('stand')
    env.render()
    time.sleep(0.03)
    
env.close()