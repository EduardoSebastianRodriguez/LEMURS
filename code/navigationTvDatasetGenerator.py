import torch
import argparse
import matplotlib.pyplot as plt
from navigationTvFunctions import realNavigation, get_cmap, generate_leader, generate_agents


def main(numTrain=1000, numTests=200, numSamples=5, seed=42, numAgents=4):

    # Set fixed random number seed
    torch.manual_seed(seed)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    simulation_time = torch.linspace(0, time-step_size, int(time/step_size))
    num_traj_train  = int(numTrain/time*step_size*numSamples)
    num_traj_tests  = int(numTests/time*step_size*numSamples)

    # Parameters
    na     = torch.as_tensor(numAgents) # Number of agents
    r      = torch.as_tensor(1.2*2.0) # Communication radius
    d      = torch.as_tensor(2.0) # Desired flocking distance
    e      = torch.as_tensor(0.1) # For the sigma-norm
    a      = torch.as_tensor(5.0) # For the sigmoid
    b      = torch.as_tensor(5.0) # For the sigmoid
    c      = torch.abs(a - b) / torch.sqrt(4 * a * b) # For the sigmoid
    c1     = torch.as_tensor(0.8)  # Tracking gain 1
    c2     = torch.as_tensor(1.0)  # Tracking gain 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = {"na": na,
                  "d": d,
                  "r": r,
                  "e": e,
                  "a": a,
                  "b": b,
                  "c": c,
                  "c1": c1,
                  "c2": c2,
                  "device": device,
                  }

    # Initialize the system to learn
    real_system = realNavigation(parameters)

    # Build training dataset
    inputsTrain   = torch.zeros(numTrain, 8 * na)
    targetTrain_2 = torch.zeros(numSamples, numTrain, 4 * na)

    print('Generating training dataset.')
    for k in range(num_traj_train):
        q_agents, p_agents               = generate_agents(na)
        q_dynamic, p_dynamic             = generate_leader(na)
        input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory                       = real_system.sample(input, simulation_time, step_size)
        l                                = int(time/step_size/numSamples)
        while torch.isnan(trajectory).any():
            q_agents, p_agents               = generate_agents(na)
            q_dynamic, p_dynamic             = generate_leader(na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = real_system.sample(input, simulation_time, step_size)
            l                                = int(time/step_size/numSamples)
        inputsTrain[l*k:l*(k+1), :]      = trajectory[::numSamples, :]
        targetTrain_2[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, numSamples, 8 * na).transpose(0, 1))[:, :, :4 * na]
        print('Navigation training instance '+str(k)+'.')
        # if k==0:
        #     plt.figure()
        #     colors = get_cmap(na)
        #     for i in range(na):
        #         plt.plot(trajectory[0, 4*na + 2*i], trajectory[0, 4*na + 2*i+1], color=colors(i), marker='x', markersize=28)
        #         plt.plot(trajectory[:, 2*i], trajectory[:, 2*i+1], color=colors(i), linewidth=2)
        #         plt.plot(trajectory[0, 2*i], trajectory[0, 2*i+1], color=colors(i), marker='o', markersize=12)
        #         plt.plot(trajectory[-1, 2*i], trajectory[-1, 2*i+1], color=colors(i), marker='s', markersize=12)
        #     plt.xlabel('x $[$m$]$')
        #     plt.ylabel('y $[$m$]$')
        #     plt.show()

    # Build test dataset
    inputsTests   = torch.zeros(numTests, 8 * na)
    targetTests_2 = torch.zeros(numSamples, numTests, 4 * na)

    print('Generating test dataset.')
    for k in range(num_traj_tests):
        q_agents, p_agents               = generate_agents(na)
        q_dynamic, p_dynamic             = generate_leader(na)
        input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory                       = real_system.sample(input, simulation_time, step_size)
        l                                = int(time/step_size/numSamples)
        while torch.isnan(trajectory).any():
            q_agents, p_agents               = generate_agents(na)
            q_dynamic, p_dynamic             = generate_leader(na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = real_system.sample(input, simulation_time, step_size)
            l                                = int(time/step_size/numSamples)
        inputsTests[l*k:l*(k+1), :]      = trajectory[::numSamples, :]
        targetTests_2[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, numSamples, 8 * na).transpose(0, 1))[:, :, :4 * na]
        print('Navigation test instance '+str(k)+'.')
        # if k==0:
        #     plt.figure()
        #     colors = get_cmap(na)
        #     for i in range(na):
        #         plt.plot(trajectory[0, 4 * na + 2 * i], trajectory[0, 4 * na + 2 * i + 1], color=colors(i), marker='x', markersize=28)
        #         plt.plot(trajectory[:, 2 * i], trajectory[:, 2 * i + 1], color=colors(i), linewidth=2)
        #         plt.plot(trajectory[0, 2 * i], trajectory[0, 2 * i + 1], color=colors(i), marker='o', markersize=12)
        #         plt.plot(trajectory[-1, 2 * i], trajectory[-1, 2 * i + 1], color=colors(i), marker='s', markersize=12)
        #     plt.xlabel('x $[$m$]$')
        #     plt.ylabel('y $[$m$]$')
        #     plt.show()

    # Store data
    torch.save(inputsTrain,   'navigationTv'+str(numAgents)+'_inputsTrain_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(targetTrain_2, 'navigationTv'+str(numAgents)+'_target2Train_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(inputsTests,   'navigationTv'+str(numAgents)+'_inputsTests_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(targetTests_2, 'navigationTv'+str(numAgents)+'_target2Tests_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset generator')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.numTrain[0], args.numTests[0], args.numSamples[0], args.seed[0], args.numAgents[0])
