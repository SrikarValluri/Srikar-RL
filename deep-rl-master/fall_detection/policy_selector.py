import gym
import gym_cassie
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision
from gym_cassie.envs.cassiemujoco import pd_in_t
from random import randint
from rl.policies import GaussianMLP
import time

state_vector = pickle.load(open("state_vector", "rb"))
print(state_vector.shape)
state_labels = np.reshape(pickle.load( open("state_labels", "rb")), (-1, 1))
print(state_labels.shape)
# s = np.arange(state_vector.shape[0])
# np.random.shuffle(s)

# state_vector = torch.from_numpy(state_vector[s])
# state_labels = torch.from_numpy(state_labels[s])

state_vector_train = torch.from_numpy(state_vector[0:1490000])
state_labels_train = torch.from_numpy(state_labels[0:1490000])

state_vector_test = torch.from_numpy(state_vector[1490000:1500000])
state_labels_test = torch.from_numpy(state_labels[1490000:1500000])

print(state_vector)
print(state_labels)

input_nodes, hidden_nodes, output_nodes, batch_size = 46, 46, 2, 100

# x = torch.randn(batch_size, input_nodes)
# y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

model = nn.Sequential(nn.Linear(input_nodes, hidden_nodes), nn.ReLU(), nn.Linear(hidden_nodes, output_nodes), nn.Sigmoid())

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    permutation = torch.randperm(state_vector_train.shape[0])
    print('Epoch: ', epoch)
    for data in range(0, state_vector_train.shape[0], batch_size):
        optimizer.zero_grad()
        # Forward Propagation
        indices = permutation[data:data+batch_size]
        batch_x, batch_y = state_vector_train[indices], state_labels_train[indices]
        outputs = model.forward(batch_x.float())    # Compute and print loss
        loss = criterion(outputs, batch_y.float())
        print('data: ', data,' loss: ', loss.item())    # Zero the gradients
        
        # perform a backward pass (backpropagation)
        loss.backward()
        
        # Update the parameters
        optimizer.step()
    pickle.dump(model, open("model.p", "wb"))







correct = 0
wrong = 0

for data in range(0, 1000):
    y_pred = model(state_vector_test[data].unsqueeze(0).float())
    if (torch.norm(model(state_vector_test[data].unsqueeze(0).float())) > 0.5 and torch.norm(state_labels_test[data].unsqueeze(0).float()) > 0.5) or (torch.norm(model(state_vector_test[data].unsqueeze(0).float())) <= 0.5 and torch.norm(state_labels_test[data].unsqueeze(0).float()) <= 0.5):
        print("correct")
        correct += 1
    else:
        print("wrong")
        wrong += 1
print(correct/(correct + wrong) * 100)

