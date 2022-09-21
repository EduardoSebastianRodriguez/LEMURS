import os
import numpy as np
import matplotlib.pyplot as plt
from Functions import get_cmap

agentsTrain = [4, 8, 16]
agentsTests = [4, 8, 16, 32, 64]
numTests    = 5

for i in range(len(agentsTrain)):
    os.system("python DatasetGenerator.py --numTrain 20000 --numTests 20000 --numSamples 5 --seed 42 --numAgents "+str(agentsTrain[i]))

for i in range(len(agentsTrain)):
    os.system("python TrainingLEMURS.py --numTrain 20000 --numTests 20000 --numSamples 5  --seed_data 42 --seed_train 42 --numAgents "+str(agentsTrain[i]))

for i in range(len(agentsTrain)):
    for j in range(len(agentsTests)):
        os.system("python Evaluation.py --seed 42 --numAgentsTrain "+str(agentsTrain[i])+" --numAgentsTests "+str(agentsTests[j])+" --numTests "+str(numTests)+" --draw 0 --seed_train 42")

losses = np.zeros([len(agentsTrain), len(agentsTests), 2])
for i in range(len(agentsTrain)):
    for j in range(len(agentsTests)):
        losses[i, j, 0] = np.load('FS_mean' + str(agentsTrain[i]) + str(agentsTests[j]) + '_loss_LEMURS.npy')
        losses[i, j, 1] = np.load('FS_std' + str(agentsTrain[i]) + str(agentsTests[j]) + '_loss_LEMURS.npy')

colors = get_cmap(len(agentsTrain))
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots()
for i in range(len(agentsTrain)):
    ax.fill_between(agentsTests, losses[i, :, 0], losses[i, :, 0] + 2 * losses[i, :, 1], color=colors(i), alpha=0.2)
    ax.plot(agentsTests, losses[i, :, 0], color=colors(i), linewidth=2, label="# train = "+str(agentsTrain[i]))
ax.set_yscale('log')
plt.xlabel('# agents')
plt.ylabel('loss')
plt.legend()
plt.show()