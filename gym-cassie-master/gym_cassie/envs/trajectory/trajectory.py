import numpy as np
import random
import torch


class CassieTrajectory:
    def __init__(self, filepath):
        n = 1 + 35 + 32 + 10 + 10 + 10
        data = np.fromfile(filepath, dtype=np.double).reshape((-1, n))
        backward_trajectory = torch.load('/home/drl/Srikar-RL/ik/trajectory/backward_stepping_traj.pkl')

        # states
        self.time = data[:, 0]
        self.qpos = data[:, 1:36]
        self.qvel = data[:, 36:68]

        # actions
        self.torque = data[:, 68:78]
        self.mpos = backward_trajectory['qpos']
        self.mvel = backward_trajectory['qvel']

    def state(self, t):
        tmax = self.time[-1]

        i = int((t % tmax) / tmax * len(self.time))

        return (self.qpos[i], self.qvel[i])

    def action(self, t):
        tmax = self.time[-1]
        i = int((t % tmax) / tmax * len(self.time))
        return (self.mpos[i], self.mvel[i], self.torque[i])

    def sample(self):
        i = random.randrange(len(self.time))
        return (self.time[i], self.qpos[i], self.qvel[i])

    def __len__(self):
        return len(self.time)