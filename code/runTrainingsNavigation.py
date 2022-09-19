import os
import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)

seeds = np.random.randint(0, 1000, 3, dtype=int)

MLP   = np.zeros([21, len(seeds)])
GNN   = np.zeros([21, len(seeds)])
GNNSA = np.zeros([21, len(seeds)])
SA    = np.zeros([21, len(seeds)])

print("-----------------------------------")
print("MLP")
print("-----------------------------------")
for i in range(len(seeds)):
    # os.system("python navigationPHMLPseeds.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed " + str(seeds[i]) + " --numAgents 4")
    MLP[:, i]   = torch.load("navigation4_" + str(seeds[i]) + '_learn_system_phmlp_TrainLosses.pth')

print("-----------------------------------")
print("GNN")
print("-----------------------------------")
for i in range(len(seeds)):
    # os.system("python navigationPHGNNseeds.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed " + str(seeds[i]) + " --numAgents 4")
    GNN[:, i]   = torch.load("navigation4_" + str(seeds[i]) + '_learn_system_phgnn_TrainLosses.pth')

print("-----------------------------------")
print("GNNSA")
print("-----------------------------------")
for i in range(len(seeds)):
    # os.system("python navigationPHGNNSAseeds.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed " + str(seeds[i]) + " --numAgents 4")
    GNNSA[:, i] = torch.load("navigation4_" + str(seeds[i]) + '_learn_system_phgnnsa_TrainLosses.pth')
GNNSA[20, :] = GNNSA[17, :]
GNNSA[19, :] = GNNSA[17, :]
GNNSA[18, :] = GNNSA[17, :]

print("-----------------------------------")
print("SA")
print("-----------------------------------")
for i in range(len(seeds)):
    # os.system("python navigationPHSAseeds.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed "+str(seeds[i])+" --numAgents 4")
    SA[:, i]    = torch.load("navigation4_" + str(seeds[i]) + '_learn_system_phsa_TrainLosses.pth')

# Plot evolution of training and validation total loss
plt.rcParams.update({'font.size': 20})
x_axis = np.linspace(0, 20 * 50, 20 + 1)
fig1, ax1 = plt.subplots()
ax1.fill_between(x_axis * 10, np.mean(MLP, axis=1), np.mean(MLP, axis=1) + 2 * np.std(MLP, axis=1), color='b', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(GNN, axis=1), np.mean(GNN, axis=1) + 2 * np.std(GNN, axis=1), color='r', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(GNNSA, axis=1), np.mean(GNNSA, axis=1) + 2 * np.std(GNNSA, axis=1), color='orange', alpha=0.2)
ax1.fill_between(x_axis * 10, np.mean(SA, axis=1), np.mean(SA, axis=1) + 2 * np.std(SA, axis=1), color='g', alpha=0.2)
ax1.plot(x_axis * 10, np.mean(MLP, axis=1), color='b', linewidth=2, label='MLP')
ax1.plot(x_axis * 10, np.mean(GNN, axis=1), color='r', linewidth=2, label='GNN')
ax1.plot(x_axis * 10, np.mean(GNNSA, axis=1), color='orange', linewidth=2, label='GNNSA')
ax1.plot(x_axis * 10, np.mean(SA, axis=1), color='g', linewidth=2, label='LEMURS')
ax1.set_yscale('log')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()