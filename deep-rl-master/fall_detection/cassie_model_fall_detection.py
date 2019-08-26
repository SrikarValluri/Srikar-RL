import gym
import gym_cassie
import numpy as np
import pickle
import torch
import time
import torchvision
import matplotlib.pyplot as plt
from gym_cassie.envs.cassiemujoco import pd_in_t
from random import randint
from rl.policies import GaussianMLP


env = gym.make("Cassie-mimic-v0")
policy_step = torch.load('/home/beast/srikar/deep-rl-master/trained_models/intel_397ts.pt')

policy_stand = torch.load('/home/beast/srikar/deep-rl-master/trained_models/standing_policy_900.pt')
model = pickle.load(open("model_trained.p", "rb" ))

def impulse(i):
    if(i == 150):
        return(1)
    else:
        return(0)

factor = 10

state = env.reset('stand')
for i in range(10 ** 10):
    env.sim.apply_force([400 * impulse(i), 0 * impulse(i), 0, 0, 0, 0])
    state = torch.Tensor(state)
    # print(model(state[0:40].unsqueeze(0).float()))
    if torch.norm(model(state[0:40].unsqueeze(0).float())) > 0.5:
        _, action = policy_step.act(torch.from_numpy(env.get_full_state('step')).float(), deterministic = True)
        state, reward, done, info = env.step(action.data.numpy(), 'step')
    else:
        _, action = policy_stand.act(torch.from_numpy(env.get_full_state('stand')[0:40]).float(), deterministic=True)
        state, reward, done, info = env.step(action.data.numpy(), 'stand')
    env.render()
    # if i == 0:
    #     input()

    reward = env.compute_reward()[0]
    plt.scatter(i, reward)
    plt.pause(0.05)
    time.sleep(0.3 / factor)
    
env.close()