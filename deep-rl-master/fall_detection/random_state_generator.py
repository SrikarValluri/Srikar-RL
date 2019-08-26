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

env = gym.make("Cassie-mimic-v0")
policy_step = torch.load('/home/beast/srikar/deep-rl-master/trained_models/walking_policy_578.pt')

random_states = []

state = env.reset('step')
end = state[46:48]
i = 0
while True:
    print(i)
    state = torch.Tensor(state)
    _, action = policy_step.act(torch.from_numpy(env.get_full_state('step')).float(), deterministic = True)
    state, reward, done, info = env.step(action.data.numpy(), 'step')
    random_states.append(state)
    if np.all(end == state[46:48]):
        break
    env.render()
    time.sleep(0.03)
    i += 1
    
env.close()

random_states = np.array(random_states)

pickle.dump(random_states, open("random_states", "wb"))