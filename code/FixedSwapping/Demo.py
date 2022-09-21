from Functions import realSystem, generate_leader, generate_agents, learnSystem, get_cmap
import matplotlib.pyplot as plt
import torch

plt.close('all')

def main():
    # Set seed
    torch.manual_seed(42)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    simulation_time = torch.linspace(0, time - step_size, int(time/step_size))

    # Parameters
    na     = torch.as_tensor(12) # Number of agents
    d      = torch.as_tensor(2.0) # Communication distance
    e      = torch.as_tensor(0.1) # For the sigma-norm
    a      = torch.as_tensor(5.0) # For the sigmoid
    b      = torch.as_tensor(5.0) # For the sigmoid
    c      = torch.abs(a - b) / torch.sqrt(4 * a * b) # For the sigmoid
    c1     = torch.as_tensor(0.8)  # Tracking gain 1
    c2     = torch.as_tensor(1.0)  # Tracking gain 2
    device = 'cpu'

    parameters = {"na": na, "d": d, "e": e, "a": a, "b": b, "c": c, "c1": c1, "c2": c2, "device": device}

    # Initialize the system to learn
    real_system       = realSystem(parameters)
    learned_system    = learnSystem(parameters)
    learned_system.load_state_dict(torch.load('FS'+str(4)+'_'+str(42)+'_learn_system_LEMURS.pth', map_location=device))
    with torch.no_grad():
        learned_system.eval()
    learned_system.na = na

    # Evaluate model
    q_agents, p_agents   = generate_agents(na)
    q_dynamic, p_dynamic = generate_leader(na)
    inputs               = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
    trajectory_real      = real_system.sample(inputs, simulation_time, step_size)
    trajectory_learned   = learned_system.forward(inputs.unsqueeze(dim=0).to(device), simulation_time, step_size).squeeze(dim=1).detach().cpu().numpy()

    # Visualize trajectories
    colors             = get_cmap(na)
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

if __name__ == '__main__':
    main()