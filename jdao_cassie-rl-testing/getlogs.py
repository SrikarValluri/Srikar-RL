import pickle
from matplotlib import pyplot as plt
import numpy as np
import cassie
import time

logs = pickle.load(open("logs/2019-07-19_15:15_log.pkl", "rb"))

states = logs["state"]
inputs= logs["input"]
times = [time.strftime('%m/%d/%Y %H:%M:%S',  time.gmtime(t)) for t in logs["time"]]

vel = []
for s in states:
    vel.append(np.linalg.norm(s.pelvis.translationalVelocity[:]))


plt.plot(vel)
plt.show()

print(times)
print(len(logs["input"]))