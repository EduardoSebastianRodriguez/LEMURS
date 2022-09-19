import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class realNavigation():

    def __init__(self, parameters):

        self.na     = parameters['na']
        self.e      = parameters['e']
        self.d      = parameters['d']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
        self.r      = parameters['r']
        self.c1     = parameters['c1']
        self.c2     = parameters['c2']

    def sigma_norm(self, z):
        return (torch.sqrt(1 + self.e * z.norm(p=2) ** 2) - 1) / self.e

    def sigmoid(self, z):
        return z / (torch.sqrt(1 + z ** 2))

    def phi_function(self, z):
        return ((self.a + self.b) * self.sigmoid(z + self.c) + (self.a - self.b)) / 2

    def phi_alpha_function(self, z, d):
        return self.phi_function(z - d)

    def f_control(self, q_agents, p_agents, q_dynamic):
        return -self.c1 * (q_agents - q_dynamic) - self.c2 * p_agents

    def laplacian(self, q_agents):
        L = torch.zeros(self.na, self.na)
        for i in range(self.na):
            for j in range(i + 1, self.na):
                L[i, j] = torch.le((q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2), self.r).float()
        L = L + L.T
        return torch.diag(torch.sum(L, 1)) - L

    def augmented_laplacian(self, q_agents):
        return torch.kron(self.laplacian(q_agents), torch.eye(2))

    def grad_V(self, L, q_agents):
        grad_V  = torch.zeros(2 * self.na)
        d_sigma = self.sigma_norm(self.d)
        for i in range(self.na):
            for j in range(self.na):
                if i != j and L[2 * i, 2 * j] != 0:
                    z_sigma = q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]
                    if self.sigma_norm(z_sigma) < d_sigma:
                        n_ij    = (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]) / (torch.sqrt(1 + self.e * (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2) ** 2))
                        grad_V[2 * i:2 * i + 2] -= z_sigma / self.sigma_norm(z_sigma) * n_ij
        return grad_V

    def flocking_dynamics(self, t, inputs):
        dq        = inputs[2 * self.na:4 * self.na] - inputs[6 * self.na:]
        dp        = -self.grad_V(self.augmented_laplacian(inputs[:2 * self.na]), inputs[:2 * self.na]) + self.f_control(inputs[:2 * self.na], inputs[2 * self.na:4 * self.na], inputs[4 * self.na:6 * self.na])
        return dq, dp

    def leader_dynamics(self, t, inputs):
        return inputs[6 * self.na:], torch.zeros(2 * self.na)

    def overall_dynamics(self, t, inputs):
        da = self.flocking_dynamics(t, inputs)
        dd = self.leader_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]))

    def sample(self, inputs, simulation_time, step_size):
        targets = odeint(self.overall_dynamics, inputs, simulation_time, method='euler', options={'step_size': step_size})
        return targets


def generate_agents():
    q_agents = torch.Tensor([4.0, 16.5, 4.0, 13.5, 4.0, 10.5, 4.0, 7.5, 4.0, 4.5, 4.0, 1.5,
                             4.0, -1.5, 4.0, -4.5, 4.0, -7.5, 4.0, -10.5, 4.0, -13.5, 4.0, -16.5,
                             -4.0, -16.5, -4.0, -13.5, -4.0, -10.5, -4.0, -7.5, -4.0, -4.5, -4.0, -1.5,
                             -4.0, 1.5, -4.0, 4.5, -4.0, 7.5, -4.0, 10.5, -4.0, 13.5, -4.0, 16.5])
    return q_agents + 0.2*torch.randn(2 * 24), torch.zeros(2 * 24)


def generate_leader():
    q_dynamic = torch.Tensor([-4.0, -16.5, -4.0, -13.5, -4.0, -10.5, -4.0, -7.5, -4.0, -4.5, -4.0, -1.5,
                             -4.0, 1.5, -4.0, 4.5, -4.0, 7.5, -4.0, 10.5, -4.0, 13.5, -4.0, 16.5,
                             4.0, 16.5, 4.0, 13.5, 4.0, 10.5, 4.0, 7.5, 4.0, 4.5, 4.0, 1.5,
                             4.0, -1.5, 4.0, -4.5, 4.0, -7.5, 4.0, -10.5, 4.0, -13.5, 4.0, -16.5])

    return q_dynamic + 0.2*torch.randn(2 * 24), torch.zeros(2 * 24)

def get_cmap(n):
    return plt.cm.get_cmap('hsv', n+1)

plt.close('all')

# Set fixed random number seed
torch.manual_seed(42)

# Hyperparameters
step_size       = 0.04
time            = 10.0
simulation_time = torch.linspace(0, time, int(time/step_size))

# Parameters
na     = torch.as_tensor(24) # Number of agents
d      = torch.as_tensor(2.0) # Desired flocking distance
r      = torch.as_tensor(1.2*2.0) # Desired flocking distance
e      = torch.as_tensor(0.1) # For the sigma-norm
a      = torch.as_tensor(5.0) # For the sigmoid
b      = torch.as_tensor(5.0) # For the sigmoid
c      = torch.abs(a - b) / torch.sqrt(4 * a * b) # For the sigmoid
c1     = torch.as_tensor(0.8) # Tracking gain 1
c2     = torch.as_tensor(1.0)  # Tracking gain 2

parameters = {"na": na,
              "d": d,
              "r": r,
              "e": e,
              "a": a,
              "b": b,
              "c": c,
              "c1": c1,
              "c2": c2,
              }

# Initialize the system to learn
real_system = realNavigation(parameters)
learned_system = torch.load('navigationTv_learn_system_phsa.pth')

print('Generating training dataset.')
q_agents, p_agents               = generate_agents()
q_dynamic, p_dynamic             = generate_leader()
inputs                           = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
trajectory_real                  = real_system.sample(inputs, simulation_time, step_size)
trajectory_learned               = learned_system.sample(inputs, simulation_time, step_size)

colors = get_cmap(na)
plt.rcParams.update({'font.size': 20})
plt.figure()
for i in range(na):
    plt.plot(trajectory[0, 4*na + 2*i], trajectory[0, 4*na + 2*i+1], color=colors(i), marker='x', markersize=28)
    plt.plot(trajectory[:, 2*i], trajectory[:, 2*i+1], color=colors(i), linewidth=2)
    plt.plot(trajectory[0, 2*i], trajectory[0, 2*i+1], color=colors(i), marker='o', markersize=12)
    plt.plot(trajectory[-1, 2*i], trajectory[-1, 2*i+1], color=colors(i), marker='s', markersize=12)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.show()

plt.figure()
for i in range(na):
    plt.plot(trajectory_learned[0, 4*na + 2*i], trajectory_learned[0, 4*na + 2*i+1], color=colors(i), marker='x', markersize=28)
    plt.plot(trajectory_learned[:, 2*i], trajectory_learned[:, 2*i+1], color=colors(i), linewidth=2)
    plt.plot(trajectory_learned[0, 2*i], trajectory_learned[0, 2*i+1], color=colors(i), marker='o', markersize=12)
    plt.plot(trajectory_learned[-1, 2*i], trajectory_learned[-1, 2*i+1], color=colors(i), marker='s', markersize=12)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.show()


epsilon_c_learned = (trajectory_learned[-1, :4 * na] - trajectory_learned[-1, 4 * na:]).norm(p=2)

errors    = (trajectory_learned[:, :4 * na] - trajectory_learned[:, 4 * na:]).norm(p=2, dim=1).le(0.05)
appear    = torch.where(errors == True)
epsilon_t_learned = appear[0]*step_size

epsilon_c_real = (trajectory_real[-1, :4 * na] - trajectory_real[-1, 4 * na:]).norm(p=2)

errors    = (trajectory_real[:, :4 * na] - trajectory_real[:, 4 * na:]).norm(p=2, dim=1).le(0.05)
appear    = torch.where(errors == True)
epsilon_t_real = appear[0]*step_size

epsilon_d = torch.sum((trajectory_real[:, :4 * na] - trajectory_learned[:, :4 * na]).norm(p=2, dim=1))

print('-------------------------------------------------')
print("Metrics to assess the difference between the trajectories from the ground-truth controller and the learned controller")
print('epsilon_c_learned = %.12f' % (epsilon_c_learned.detach().cpu().numpy()))
print('epsilon_c_real = %.12f' % (epsilon_c_real.detach().cpu().numpy()))
print('epsilon_t_learned = %.12f' % (epsilon_t_learned.detach().cpu().numpy()))
print('epsilon_t_real = %.12f' % (epsilon_t_real.detach().cpu().numpy()))
print('epsilon_d = %.12f' % (epsilon_d.detach().cpu().numpy()))
print('-------------------------------------------------')