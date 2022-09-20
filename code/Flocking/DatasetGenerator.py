import torch
import argparse
from Functions import realSystem, generate_leader, generate_agents


def main(numTrain=1000, numTests=200, numSamples=5, seed=42, numAgents=4):

    # Set seed
    torch.manual_seed(seed)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    simulation_time = torch.linspace(0, time-step_size, int(time/step_size))
    num_traj_train  = int(numTrain/time*step_size*numSamples)
    num_traj_tests  = int(numTests/time*step_size*numSamples)

    # Parameters
    na = torch.as_tensor(numAgents)  # Number of agents
    d = torch.as_tensor(1.0)  # Desired flocking distance
    r = torch.as_tensor(1.2) * d  # Radius of influence
    e = torch.as_tensor(0.1)  # For the sigma-norm
    a = torch.as_tensor(5.0)  # For the sigmoid
    b = torch.as_tensor(5.0)  # For the sigmoid
    c = torch.abs(a - b) / torch.sqrt(4 * a * b)  # For the sigmoid
    ha = torch.as_tensor(0.2)  # For the sigmoid
    c1 = torch.as_tensor(0.4)  # Tracking gain 1
    c2 = torch.as_tensor(0.8)  # Tracking gain 2

    parameters = {"na": na, "d": d, "r": r, "e": e, "a": a, "b": b, "c": c, "ha": ha, "c1": c1, "c2": c2}

    # Initialize the system to learn
    real_system = realSystem(parameters)

    # Build training dataset
    inputs_train = torch.zeros(numTrain, 8 * na)
    target_train = torch.zeros(numSamples, numTrain, 4 * na)

    print('Generating training dataset.')
    for k in range(num_traj_train):
        q_agents, p_agents               = generate_agents(na)
        q_dynamic, p_dynamic             = generate_leader(na)
        input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory                       = real_system.sample(input, simulation_time, step_size)
        l                                = int(time/step_size/numSamples)

        # Avoid strange examples due to numerical or initial configuration issues
        while torch.isnan(trajectory).any():
            q_agents, p_agents               = generate_agents(na)
            q_dynamic, p_dynamic             = generate_leader(na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = real_system.sample(input, simulation_time, step_size)
            l                                = int(time/step_size/numSamples)
        inputs_train[l*k:l*(k+1), :]    = trajectory[::numSamples, :]
        target_train[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, numSamples, 8 * na).transpose(0, 1))[:, :, :4 * na]
        print('Training instance '+str(k)+'.')

    # Build test dataset
    inputs_tests = torch.zeros(numTests, 8 * na)
    target_tests = torch.zeros(numSamples, numTests, 4 * na)

    print('Generating test dataset.')
    for k in range(num_traj_tests):
        q_agents, p_agents               = generate_agents(na)
        q_dynamic, p_dynamic             = generate_leader(na)
        input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
        trajectory                       = real_system.sample(input, simulation_time, step_size)
        l                                = int(time/step_size/numSamples)

        # Avoid strange examples due to numerical or initial configuration issues
        while torch.isnan(trajectory).any():
            q_agents, p_agents               = generate_agents(na)
            q_dynamic, p_dynamic             = generate_leader(na)
            input                            = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
            trajectory                       = real_system.sample(input, simulation_time, step_size)
            l                                = int(time/step_size/numSamples)
        inputs_tests[l*k:l*(k+1), :]    = trajectory[::numSamples, :]
        target_tests[:, l*k:l*(k+1), :] = (trajectory.reshape(-1, numSamples, 8 * na).transpose(0, 1))[:, :, :4 * na]
        print('Test instance '+str(k)+'.')

    # Store data
    torch.save(inputs_train, 'F'+str(numAgents)+'_inputsTrain_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(target_train, 'F'+str(numAgents)+'_targetTrain_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(inputs_tests, 'F'+str(numAgents)+'_inputsTests_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')
    torch.save(target_tests, 'F'+str(numAgents)+'_targetTests_'+str(numTrain)+str(numTests)+str(numSamples)+str(seed)+'_.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset generator for the flocking problem')
    parser.add_argument('--numTrain', type=int, nargs=1, help='number of instances for training')
    parser.add_argument('--numTests', type=int, nargs=1, help='number of instances for testing')
    parser.add_argument('--numSamples', type=int, nargs=1, help='number of samples per instance')
    parser.add_argument('--seed', type=int, nargs=1, help='seed to generate reproducible data')
    parser.add_argument('--numAgents', type=int, nargs=1, help='number of agents')
    args = parser.parse_args()
    main(args.numTrain[0], args.numTests[0], args.numSamples[0], args.seed[0], args.numAgents[0])
