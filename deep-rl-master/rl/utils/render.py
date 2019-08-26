import torch
from torch.autograd import Variable
import time

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset())
    i = 45
    while True:
        _, action = policy.act(state, deterministic)
        
        state, reward, done, _ = env.step(action[0].data.numpy())

        if done or i > 45:
            state = env.reset()
            i = -1

        state = torch.Tensor(state)

        env.render()

        if i == 0:
            input()

        time.sleep(dt / speedup)

        i += 1

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)