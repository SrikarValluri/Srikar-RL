import gym
import gym_cassie
import numpy as np
import pickle
import torch
import torchvision
from gym_cassie.envs.cassiemujoco import pd_in_t
from random import randint
from rl.policies import GaussianMLP


env = gym.make("Cassie-mimic-v0")
policy_step = torch.load('/home/beast/srikar/deep-rl-master/trained_models/intel_397ts.pt')
policy_stand = torch.load('/home/beast/srikar/deep-rl-master/trained_models/standing_policy_900.pt')
model = pickle.load(open("model.p", "rb" ))

def impulse(i):
    if(i == 30):
        return(1)
    else:
        return(0)

state = env.reset()
for i in range(10000):
    env.sim.apply_force([500 * impulse(i), 0, 0, 0, 0, 0])
    state = torch.Tensor(state)
    if model(state[0:40]) == 1:
        _, action = policy_step.act(state, deterministic = True)
    else:
        _, action = policy_stand.act(state[0:40], deterministic=True)
    state, reward, done, info = env.step(action.data.numpy())
    env.render()
    
env.close()