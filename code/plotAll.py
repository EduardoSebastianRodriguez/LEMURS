import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(example, numAgents):

    # Log train and evaluation loss
    TrainLossesMLP = torch.load(example +str(numAgents)+ '_learn_system_phmlp_TrainLosses.pth')
    TestsLossesMLP = torch.load(example +str(numAgents)+ '_learn_system_phmlp_TestsLosses.pth')
    TrainLossesGNN = torch.load(example +str(numAgents)+ '_learn_system_phgnn_TrainLosses.pth')
    TestsLossesGNN = torch.load(example +str(numAgents)+ '_learn_system_phgnn_TestsLosses.pth')
    TrainLossesSA = torch.load(example +str(numAgents)+ '_learn_system_phsa_TrainLosses.pth')
    TestsLossesSA = torch.load(example +str(numAgents)+ '_learn_system_phsa_TestsLosses.pth')
    TrainLossesGNNSA = torch.load(example +str(numAgents)+ '_learn_system_phgnnsa_TrainLosses.pth')
    TestsLossesGNNSA = torch.load(example +str(numAgents)+ '_learn_system_phgnnsa_TestsLosses.pth')

    # Plot evolution of training and validation total loss
    plt.rcParams.update({'font.size': 20})
    x_axis = np.linspace(0, len(TestsLossesMLP) * 50, len(TestsLossesMLP) + 1)[:-1]
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis*10, TrainLossesMLP, 'b', linewidth=3, label='MLP')
    # ax1.plot(x_axis*10, TestsLossesMLP, 'c', linewidth=3, label='Test loss MLP')
    ax1.plot(x_axis*10, TrainLossesGNN, 'r', linewidth=3, label='GNN')
    # ax1.plot(x_axis*10, TestsLossesGNN, 'lightcoral', linewidth=3, label='Test loss GNN')
    ax1.plot(x_axis*10, TrainLossesGNNSA, 'orange', linewidth=3, label='GNNSA')
    # ax1.plot(x_axis*10, TestsLossesGNNSA, 'bisque', linewidth=3, label='Test loss GNNSA')
    ax1.plot(x_axis*10, TrainLossesSA[:41], 'g', linewidth=3, label='SA (Ours)')
    # ax1.plot(x_axis*10, TestsLossesSA[:41], 'palegreen', linewidth=3, label='Test loss SA')
    ax1.set_yscale('log')
    ax1.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot all solutions')
    parser.add_argument('--example', type=str, nargs=1, help='example')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.example[0], args.numAgents[0])