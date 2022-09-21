from Functions import realSystem, generate_leader, learnSystem, generate_agents
import matplotlib.pyplot as plt
import torch

plt.close('all')

def main():

    # Set seed
    torch.manual_seed(42)

    # Hyperparameters
    step_size       = 0.04
    time            = 10.0
    simulation_time = torch.linspace(0, time, int(time/step_size))

    # Parameters
    na = torch.as_tensor(12)  # Number of agents
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

    parameters = {"na": na, "d": d, "r": r, "e": e, "a": a, "b": b, "c": c, "ha": ha, "c1": c1, "c2": c2, "device": device}

    # Initialize the system to learn
    real_system       = realSystem(parameters)
    learned_system    = learnSystem(parameters)
    learned_system.load_state_dict(torch.load('F'+str(4)+'_'+str(42)+'_learn_system_LEMURS.pth', map_location=device))
    with torch.inference_mode():
        learned_system.eval()
    learned_system.na = na

    # Evaluate model
    q_agents, p_agents               = generate_agents(na)
    q_dynamic, p_dynamic             = generate_leader(na)
    inputs                           = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
    trajectory_real                  = real_system.sample(inputs, simulation_time, step_size)
    trajectory_learned               = learned_system.forward(inputs.unsqueeze(dim=0).to(device), simulation_time, step_size).squeeze(dim=1)

    # Visualize trajectories
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
                        [trajectory_learned[-1, 2 * i + 1], trajectory_learned[-1, 2 * j + 1]], 'b', linewidth=3.0)
    for i in range(na):
        ax.plot(trajectory_learned[:, 2 * i], trajectory_learned[:, 2 * i + 1], 'k')
        ax.plot(trajectory_learned[0, 2 * i], trajectory_learned[0, 2 * i + 1], 'go', markersize=14)
        ax.plot(trajectory_learned[-1, 2 * i], trajectory_learned[-1, 2 * i + 1], 'bs', markersize=14)
    ax.plot(trajectory_learned[0, 4 * na], trajectory_learned[0, 4 * na + 1], 'rx', markersize=24)
    plt.xlabel('x $[$m$]$')
    plt.ylabel('y $[$m$]$')
    plt.show()

if __name__ == '__main__':
    main()