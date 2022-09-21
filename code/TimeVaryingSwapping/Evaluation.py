from Functions import realSystem, generate_leader, generate_agents, learnSystem, get_cmap
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse

plt.close('all')

def main(seed=42, numAgentsTrain=4, numAgentsTests=4, numTests=20, draw=0, seed_train=42):

    # Set seed
    torch.manual_seed(seed)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    nt              = numTests
    simulation_time = torch.linspace(0, time, int(time/step_size))

    # Parameters
    na     = torch.as_tensor(numAgentsTests) # Number of agents
    d      = torch.as_tensor(2.0) # Desired flocking distance
    r      = torch.as_tensor(1.2*2.0) # Communication radius
    e      = torch.as_tensor(0.1) # For the sigma-norm
    a      = torch.as_tensor(5.0) # For the sigmoid
    b      = torch.as_tensor(5.0) # For the sigmoid
    c      = torch.abs(a - b) / torch.sqrt(4 * a * b) # For the sigmoid
    c1     = torch.as_tensor(0.8)  # Tracking gain 1
    c2     = torch.as_tensor(1.0)  # Tracking gain 2
    device = 'cpu'

    parameters = {"na": na, "d": d, "r": r, "e": e, "a": a, "b": b, "c": c, "c1": c1, "c2": c2, "device": device}

    # Initialize the system to learn
    real_system       = realSystem(parameters)
    learned_system    = learnSystem(parameters)
    learned_system.load_state_dict(torch.load('TVS'+str(numAgentsTrain)+'_'+str(seed_train)+'_learn_system_LEMURS.pth', map_location=device))
    with torch.no_grad():
        learned_system.eval()
    learned_system.na = na

    # Evaluate model
    loss = np.zeros(nt)
    for test in range(nt):
        q_agents, p_agents               = generate_agents(na)
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
            trajectory_learned = trajectory_learned.detach().cpu().numpy()
            colors = get_cmap(na)
            plt.rcParams.update({'font.size': 20})
            plt.figure()
            for i in range(na):
                plt.plot(trajectory_real[0, 4 * na + 2 * i], trajectory_real[0, 4 * na + 2 * i + 1], color=colors(i), marker='x', markersize=28)
                plt.plot(trajectory_real[:, 2 * i], trajectory_real[:, 2 * i + 1], color=colors(i), linewidth=2)
                plt.plot(trajectory_real[0, 2 * i], trajectory_real[0, 2 * i + 1], color=colors(i), marker='o', markersize=12)
                plt.plot(trajectory_real[-1, 2 * i], trajectory_real[-1, 2 * i + 1], color=colors(i), marker='s', markersize=12)
            plt.xlabel('x $[$m$]$')
            plt.ylabel('y $[$m$]$')
            plt.show()

            plt.figure()
            for i in range(na):
                plt.plot(trajectory_learned[0, 4 * na + 2 * i], trajectory_learned[0, 4 * na + 2 * i + 1], color=colors(i), marker='x', markersize=28)
                plt.plot(trajectory_learned[:, 2 * i], trajectory_learned[:, 2 * i + 1], color=colors(i), linewidth=2)
                plt.plot(trajectory_learned[0, 2 * i], trajectory_learned[0, 2 * i + 1], color=colors(i), marker='o', markersize=12)
                plt.plot(trajectory_learned[-1, 2 * i], trajectory_learned[-1, 2 * i + 1], color=colors(i), marker='s', markersize=12)
            plt.xlabel('x $[$m$]$')
            plt.ylabel('y $[$m$]$')
            plt.show()

    # Normalize loss wrt the size of the stage
    loss_mean = np.mean(loss) / (6.0**2 + (3.0*int(na/2))**2)
    loss_std  = np.std(loss) / (6.0**2 + (3.0*int(na/2))**2)

    # Print and save
    print('Loss is of mean %.12f and std %.12f' % (loss_mean, loss_std))
    np.save('TVS_mean' + str(numAgentsTrain) + str(numAgentsTests) + '_loss_LEMURS.npy', loss_mean)
    np.save('TVS_std' + str(numAgentsTrain) + str(numAgentsTests) + '_loss_LEMURS.npy', loss_std)
    np.save('TVS_trajectory_real' + str(numAgentsTrain) + str(numAgentsTests) + '_LEMURS.npy', trajectory_real)
    np.save('TVS_trajectory_learned' + str(numAgentsTrain) + str(numAgentsTests) + '_LEMURS.npy', trajectory_learned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-varying swapping evaluation')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgentsTrain', type=int, nargs=1, help='number of agents')
    parser.add_argument('--numAgentsTests', type=int, nargs=1, help='number of agents')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of tests')
    parser.add_argument('--draw', type=int, nargs=1, help='do you want to display the trajectories? --- 1 is True, 0 is False')
    parser.add_argument('--seed_train', type=int, nargs=1, help='seed used in training')
    args = parser.parse_args()
    main(args.seed[0], args.numAgentsTrain[0], args.numAgentsTests[0], args.numTests[0], args.draw[0], args.seed_train[0])