from Functions import realSystem, generate_leader, learnSystem, generate_agents_test
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.close('all')

def main(seed=42, numAgentsTrain=4, numAgentsTests=4, numTests=20, draw=False):

    # Set seed
    torch.manual_seed(seed)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    nt              = numTests
    simulation_time = torch.linspace(0, time, int(time/step_size))

    # Parameters
    na = torch.as_tensor(numAgentsTests)  # Number of agents
    d = torch.as_tensor(1.0)  # Desired flocking distance
    r = torch.as_tensor(1.2) * d  # Radius of influence
    e = torch.as_tensor(0.1)  # For the sigma-norm
    a = torch.as_tensor(5.0)  # For the sigmoid
    b = torch.as_tensor(5.0)  # For the sigmoid
    c = torch.abs(a - b) / torch.sqrt(4 * a * b)  # For the sigmoid
    ha = torch.as_tensor(0.2)  # For the sigmoid
    c1 = torch.as_tensor(0.4)  # Tracking gain 1
    c2 = torch.as_tensor(0.8)  # Tracking gain 2
    device = 'cpu'

    parameters = {"na": na, "d": d, "r": r, "e": e, "a": a, "b": b, "c": c, "ha": ha, "c1": c1, "c2": c2}

    # Initialize the system to learn
    real_system       = realSystem(parameters)
    learned_system    = learnSystem(parameters)
    learned_system.load_state_dict(torch.load('F'+str(numAgentsTrain)+'_learn_system_LEMURS.pth', map_location=device))
    with torch.inference_mode():
        learned_system.eval()
    learned_system.na = na

    # Evaluate model
    loss = np.zeros(nt)
    for test in range(nt):
        q_agents, p_agents               = generate_agents_test(na)
        q_dynamic, p_dynamic             = generate_leader(na)
        inputs                           = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory_real                  = real_system.sample(inputs, simulation_time, step_size)
        trajectory_learned               = torch.zeros(int(time/step_size), 8 * na, device=device)

        # To avoid problems with the memory space and garbage collector
        for t in range(int(time/step_size/5)):
            tray                     = learned_system.assess(inputs.unsqueeze(dim=0).to(device), simulation_time[t:t+5], step_size).squeeze(dim=1)
            trajectory_learned[5*t:5*t+5, :] = tray
            inputs                   = tray[-1, :]

        # Compute loss
        t_real     = np.concatenate((trajectory_real[:, 0:2 * na:2].detach().cpu().numpy(), trajectory_real[:, 1:2 * na:2].detach().cpu().numpy(), trajectory_real[:, 2 * na:4 * na:2].detach().cpu().numpy(), trajectory_real[:, 2 * na+1:4 * na:2].detach().cpu().numpy()), axis=1)
        t_real     = np.reshape(t_real, [-1, na, 4])
        t_learned  = np.concatenate((trajectory_learned[:, 0:2 * na:2].detach().cpu().numpy(), trajectory_learned[:, 1:2 * na:2].detach().cpu().numpy(), trajectory_learned[:, 2 * na:4 * na:2].detach().cpu().numpy(), trajectory_learned[:, 2 * na+1:4 * na:2].detach().cpu().numpy()), axis=1)
        t_learned  = np.reshape(t_learned, [-1, na, 4])
        loss[test] = np.mean(np.mean(np.linalg.norm((t_real - t_learned), axis=2) ** 2, axis=1))

        # Visualize trajectories
        if draw:
            plt.rcParams.update({'font.size': 20})
            laplacian = real_system.laplacian(real_system.ha, trajectory_real[-1, : 2 * na])
            trajectory_real = trajectory_real.detach().cpu().numpy()
            fig, ax = plt.subplots()
            for i in range(na):
                for j in range(i + 1, na):
                    if laplacian[i, j] != 0:
                        ax.plot([trajectory_real[-1, 2 * i], trajectory_real[-1, 2 * j]],
                                [trajectory_real[-1, 2 * i + 1], trajectory_real[-1, 2 * j + 1]], 'b', linewidth=3)
            for i in range(na):
                ax.plot(trajectory_real[:, 2 * i], trajectory_real[:, 2 * i + 1], 'k')
                ax.plot(trajectory_real[0, 2 * i], trajectory_real[0, 2 * i + 1], 'go', markersize=14)
                ax.plot(trajectory_real[-1, 2 * i], trajectory_real[-1, 2 * i + 1], 'bs', markersize=14)
            ax.plot(trajectory_real[0, 4 * na], trajectory_real[0, 4 * na + 1], 'rx', markersize=24)
            plt.xlabel('x $[$m$]$')
            plt.ylabel('y $[$m$]$')
            plt.show()

            laplacian = real_system.laplacian(real_system.ha, trajectory_learned[-1, :2 * na])
            trajectory_learned = trajectory_learned.detach().cpu().numpy()
            fig, ax = plt.subplots()
            for i in range(na):
                for j in range(i + 1, na):
                    if laplacian[i, j] != 0:
                        ax.plot([trajectory_learned[-1, 2 * i], trajectory_learned[-1, 2 * j]],
                                [trajectory_learned[-1, 2 * i + 1], trajectory_learned[-1, 2 * j + 1]], 'b', linewidth=3)
            for i in range(na):
                ax.plot(trajectory_learned[:, 2 * i], trajectory_learned[:, 2 * i + 1], 'k')
                ax.plot(trajectory_learned[0, 2 * i], trajectory_learned[0, 2 * i + 1], 'go', markersize=14)
                ax.plot(trajectory_learned[-1, 2 * i], trajectory_learned[-1, 2 * i + 1], 'bs', markersize=14)
            ax.plot(trajectory_learned[0, 4 * na], trajectory_learned[0, 4 * na + 1], 'rx', markersize=24)
            plt.xlabel('x $[$m$]$')
            plt.ylabel('y $[$m$]$')
            plt.show()

    # Normalize loss wrt the size of the stage
    loss_mean = np.mean(loss) / (np.sqrt(numAgentsTests + numAgentsTests) * d.numpy())
    loss_std  = np.std(loss) / (np.sqrt(numAgentsTests + numAgentsTests) * d.numpy())

    # Print and save
    print('Loss is of mean %.12f and std %.12f' % (loss_mean, loss_std))
    np.save('F_mean' + str(numAgentsTrain) + str(numAgentsTests) + '_loss_LEMURS.npy', loss_mean)
    np.save('F_std' + str(numAgentsTrain) + str(numAgentsTests) + '_loss_LEMURS.npy', loss_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flocking evaluation')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgentsTrain', type=int, nargs=1, help='number of agents')
    parser.add_argument('--numAgentsTests', type=int, nargs=1, help='number of agents')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of tests')
    parser.add_argument('--draw', type=int, nargs=1, help='do you want to display the trajectories? --- 1 is True, 0 is False')
    args = parser.parse_args()
    main(args.seed[0], args.numAgentsTrain[0], args.numAgentsTests[0], args.numTests[0], args.draw[0])