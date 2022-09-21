import os
import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)

seeds     = np.random.randint(0, 1000, 3, dtype=int)
seed_data = 42
na        = 4

MLP   = np.zeros([81, len(seeds)])
GNN   = np.zeros([81, len(seeds)])
GNNSA = np.zeros([81, len(seeds)])
LEMURS= np.zeros([81, len(seeds)])

print("-----------------------------------")
print("MLP")
print("-----------------------------------")
for i in range(len(seeds)):
    os.system("python TrainingPHMLP.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed_data "+str(seed_data)+" --seed_train " + str(seeds[i])+" --numAgents "+str(na))
    MLP[:, i]   = torch.load("TVS"+str(na)+"_" + str(seeds[i]) + '_learn_system_phmlp_TrainLosses.pth')

print("-----------------------------------")
print("GNN")
print("-----------------------------------")
for i in range(len(seeds)):
    os.system("python TrainingPHGNN.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed_data "+str(seed_data)+" --seed_train " + str(seeds[i])+" --numAgents "+str(na))
    GNN[:, i]   = torch.load("TVS"+str(na)+"_" + str(seeds[i]) + '_learn_system_phgnn_TrainLosses.pth')

print("-----------------------------------")
print("GNNSA")
print("-----------------------------------")
for i in range(len(seeds)):
    os.system("python TrainingPHGNNSA.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed_data "+str(seed_data)+" --seed_train " + str(seeds[i])+" --numAgents "+str(na))
    GNNSA[:, i] = torch.load("TVS"+str(na)+"_" + str(seeds[i]) + '_learn_system_phgnnsa_TrainLosses.pth')

print("-----------------------------------")
print("LEMURS")
print("-----------------------------------")
for i in range(len(seeds)):
    os.system("python TrainingLEMURS.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed_data "+str(seed_data)+" --seed_train " + str(seeds[i])+" --numAgents "+str(na))
    LEMURS[:, i]    = torch.load("TVS"+str(na)+"_" + str(seeds[i]) + '_learn_system_LEMURS_TrainLosses.pth')

# Plot evolution of training and validation total loss
plt.rcParams.update({'font.size': 20})
x_axis = np.linspace(0, 20 * 50, 20 + 1)
fig1, ax1 = plt.subplots()
ax1.fill_between(x_axis * 10, np.mean(MLP[:21, :], axis=1), np.mean(MLP[:21, :], axis=1) + 2 * np.std(MLP[:21, :], axis=1), color='b', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(GNN[:21, :], axis=1), np.mean(GNN[:21, :], axis=1) + 2 * np.std(GNN[:21, :], axis=1), color='r', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(GNNSA[:21, :], axis=1), np.mean(GNNSA[:21, :], axis=1) + 2 * np.std(GNNSA[:21, :], axis=1), color='orange', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(LEMURS[:21, :], axis=1), np.mean(LEMURS[:21, :], axis=1) + 2 * np.std(LEMURS[:21, :], axis=1), color='g', alpha=0.2)
ax1.plot(x_axis * 10, np.mean(MLP[:21, :], axis=1), color='b', linewidth=2, label='MLP')
ax1.plot(x_axis * 10, np.mean(GNN[:21, :], axis=1), color='r', linewidth=2, label='GNN')
ax1.plot(x_axis * 10, np.mean(GNNSA[:21, :], axis=1), color='orange', linewidth=2, label='GNNSA')
ax1.plot(x_axis * 10, np.mean(LEMURS[:21, :], axis=1), color='g', linewidth=2, label='LEMURS')
ax1.set_yscale('log')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()