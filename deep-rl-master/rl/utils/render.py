import torch
from torch.autograd import Variable
import time

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset())
    r_total = 0
    i = 0
    while True:
        _, action = policy.act(state, True)

        state, reward, done, _ = env.step(action[0].data.numpy())

        state = torch.Tensor(state)

        env.render()

        time.sleep(dt / speedup)

        i += 1
        r_total += reward

        if done:
            print("Reward: ", r_total)
            break

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)