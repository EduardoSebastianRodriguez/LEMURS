import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class realFlocking():

    def __init__(self, parameters):

        self.na     = parameters['na']
        self.nb     = parameters['nb']
        self.d      = parameters['d']
        self.r      = parameters['r']
        self.e      = parameters['e']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
        self.ha     = parameters['ha']
        self.hb     = parameters['hb']
        self.varp   = parameters['varp']
        self.vMax   = parameters['vMax']
        self.vMin   = parameters['vMin']
        self.c1     = parameters['c1']
        self.c2     = parameters['c2']
        self.oMax   = parameters['oMax']
        self.oMin   = parameters['oMin']
        self.gap    = parameters['gap']
        self.static = parameters['static']

    def sigma_norm(self, z):
        return (torch.sqrt(1 + self.e * z.norm(p=2) ** 2) - 1) / self.e

    def rho_function(self, z, h):
        if 0 <= z < h:
            return 1
        elif h <= z <= 1:
            pi = torch.acos(torch.zeros(1)).item() * 2
            return (1 + torch.cos(pi * ((z - h) / (1 - h)))) / 2
        else:
            return 0

    def sigmoid(self, z):
        return z / (torch.sqrt(1 + z ** 2))

    def phi_function(self, z):
        return ((self.a + self.b) * self.sigmoid(z + self.c) + (self.a - self.b)) / 2

    def phi_alpha_function(self, z, r, h, d):
        return self.rho_function(z / r, h) * self.phi_function(z - d)

    def f_control(self, q_agents, p_agents, q_dynamic, p_dynamic):
        qr_expanded = torch.zeros(2 * self.na)
        pr_expanded = torch.zeros(2 * self.na)
        for i in range(self.na):
            qr_expanded[2 * i:2 * i + 2] = q_dynamic
            pr_expanded[2 * i:2 * i + 2] = p_dynamic

        return -self.c1 * (q_agents - qr_expanded) - self.c2 * (p_agents - pr_expanded)

    def laplacian(self, h, q_agents):
        L       = torch.zeros(self.na, self.na)
        r_sigma = self.sigma_norm(self.r)
        for i in range(self.na):
            for j in range(i + 1, self.na):
                z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2])
                L[i, j] = self.rho_function(z_sigma / r_sigma, h)
        L = L + L.T
        return torch.diag(torch.sum(L, 1)) - L

    def augmented_laplacian(self, h, q_agents):
        return torch.kron(self.laplacian(h, q_agents), torch.eye(2))

    def getLaplacians(self, h, q_agents):
        L       = torch.zeros(q_agents.shape[0], self.na, self.na)
        r_sigma = self.sigma_norm(self.r)
        for k in range(q_agents.shape[0]):
            for i in range(self.na):
                for j in range(i + 1, self.na):
                    z_sigma = self.sigma_norm(q_agents[k, 2 * j:2 * j + 2] - q_agents[k, 2 * i:2 * i + 2])
                    value   = self.rho_function(z_sigma / r_sigma, h)
                    if value != 0:
                        L[k, i, j] = 1
                    else:
                        L[k, i, j] = 0
            L[k, :, :] = L[k, :, :] + torch.transpose(L[k, :, :], 0, 1)
            L[k, :, :] = torch.diag(torch.sum(L[k, :, :], 1)) - L[k, :, :]
        return L

    def grad_V(self, h, L, q_agents):
        grad_V  = torch.zeros(2 * self.na)
        r_sigma = self.sigma_norm(self.r)
        d_sigma = self.sigma_norm(self.d)
        for i in range(self.na):
            for j in range(self.na):
                if i != j and L[2 * i, 2 * j] != 0:
                    z_sigma = self.sigma_norm(q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2])
                    n_ij    = (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]) / (torch.sqrt(1 + self.e * (q_agents[2 * j:2 * j + 2] - q_agents[2 * i:2 * i + 2]).norm(p=2) ** 2))
                    grad_V[2 * i:2 * i + 2] -= self.phi_alpha_function(z_sigma, r_sigma, h, d_sigma) * n_ij
        return grad_V

    def grad_Vo(self, h, q_obstac, q_agents):
        grad_Vo = torch.zeros(2 * self.na)
        r_sigma = self.sigma_norm(self.r)
        d_sigma = self.sigma_norm(self.d)
        for i in range(self.na):
            for j in range(self.nb):
                mu_i      = q_obstac[3 * j + 0] / (q_agents[2 * i:2 * i + 2] - q_obstac[3 * j + 1:3 * j + 3]).norm(p=2)
                qi_obstac = mu_i * q_agents[2 * i:2 * i + 2] + (1 - mu_i) * q_obstac[3 * j + 1:3 * j + 3]
                z_sigma   = self.sigma_norm(q_agents[2 * i:2 * i + 2] - qi_obstac)
                n_ij      = (qi_obstac - q_agents[2 * i:2 * i + 2]) / torch.sqrt(1 + self.e * (q_agents[2 * i:2 * i + 2] - qi_obstac).norm(p=2) ** 2)
                grad_Vo[2 * i:2 * i + 2] -= self.phi_alpha_function(z_sigma, r_sigma, h, d_sigma) * n_ij
        return grad_Vo

    def flocking_dynamics(self, t, inputs):

        L         = self.augmented_laplacian(self.ha, inputs[3 * self.nb:3 * self.nb + 2 * self.na])
        DeltaV    = self.grad_V(self.ha, L, inputs[3 * self.nb:3 * self.nb + 2 * self.na])
        DeltaVo   = self.grad_Vo(self.hb, inputs[0:3 * self.nb], inputs[3 * self.nb:3 * self.nb + 2 * self.na])

        dq        = inputs[3 * self.nb + 2 * self.na:3 * self.nb + 4 * self.na]
        dp        = -DeltaV \
                    -L @ inputs[3 * self.nb + 2 * self.na:3 * self.nb + 4 * self.na] \
                    +self.f_control(inputs[3 * self.nb:3 * self.nb + 2 * self.na],
                                    inputs[3 * self.nb + 2 * self.na:3 * self.nb + 4 * self.na],
                                    inputs[3 * self.nb + 4 * self.na:3 * self.nb + 4 * self.na + 2],
                                    inputs[3 * self.nb + 4 * self.na + 2:]) \
                    -DeltaVo

        return dq, dp

    def leader_dynamics(self, t, inputs):
        return inputs[3 * self.nb + 4 * self.na + 2:], torch.zeros(2)

    def overall_dynamics(self, t, inputs):
        da = self.flocking_dynamics(t, inputs)
        dd = self.leader_dynamics(t, inputs)
        return torch.cat((torch.zeros(3 * self.nb), da[0], da[1], dd[0], dd[1]))

    def sample(self, inputs, simulation_time, step_size):
        targets = odeint(self.overall_dynamics, inputs, simulation_time, method='euler', options={'step_size': step_size})
        return targets


def generate_obstacles(nb, oMax, oMin, varp, gap):
    q_obstac = torch.zeros(3 * nb)
    for i in range(nb):
        q_obstac[3 * i + 0] = (oMax - oMin) * torch.rand(1) + oMin
    for i in range(nb):
        if i == 0:
            q_obstac[3 * i + 1] = torch.normal(0, varp)
            q_obstac[3 * i + 2] = torch.normal(0, varp)
        else:
            ok = 0
            while ok == 0:
                ok = 1
                q_obstac[3 * i + 1] = torch.normal(0, varp)
                q_obstac[3 * i + 2] = torch.normal(0, varp)
                for j in range(0, i):
                    if (q_obstac[3 * i + 1:3 * i + 3] - q_obstac[3 * j + 1:3 * j + 3]).norm(p=2) < q_obstac[3 * j + 0] + q_obstac[3 * i + 0] + 2 * gap:
                        ok = 0
    return q_obstac


def generate_agents(na, varp, vMax, vMin, nb, q_obstac, gap):
    q_agents = torch.zeros(2 * na)
    p_agents = torch.zeros(2 * na)
    for i in range(na):
        ok = 0
        while ok == 0:
            ok = 1
            q_agents[2 * i:2 * i + 2] = torch.Tensor([torch.normal(0, varp), torch.normal(0, varp)])
            p_agents[2 * i:2 * i + 2] = (vMax - vMin) * torch.rand(2) + vMin
            for j in range(nb):
                if (q_obstac[3 * j + 1:3 * j + 3] - q_agents[2 * i:2 * i + 2]).norm(p=2) < q_obstac[3 * j + 0] + gap:
                    ok = 0
    return q_agents, p_agents


def generate_leader(varp, nb, q_obstac, gap, static, vMax, vMin):
    q_dynamic = torch.zeros(2)
    ok = 0
    while ok == 0:
        ok = 1
        q_dynamic = torch.Tensor([torch.normal(0, varp), torch.normal(0, varp)])
        for j in range(nb):
            if (q_obstac[3 * j + 1:3 * j + 3] - q_dynamic).norm(p=2) < q_obstac[3 * j + 0] + gap:
                ok = 0
    if static == 1:
        p_dynamic = torch.zeros(2)
    else:
        p_dynamic = (vMax - vMin) * torch.rand(2) + vMin

    return torch.zeros(2), torch.zeros(2)

plt.close('all')

# Hyperparameters
step_size       = 0.04
time            = 15.0
simulation_time = torch.linspace(0, time, int(time/step_size))

# Parameters
na     = torch.as_tensor(30) # Number of agents
nb     = torch.as_tensor(0) # Number of obstacles
d      = torch.as_tensor(2.0) # Desired flocking distance
r      = torch.as_tensor(1.2)*d # Radius of influence
e      = torch.as_tensor(0.1) # For the sigma-norm
a      = torch.as_tensor(5.0) # For the sigmoid
b      = torch.as_tensor(5.0) # For the sigmoid
c      = torch.abs(a - b) / torch.sqrt(4 * a * b) # For the sigmoid
ha     = torch.as_tensor(0.2) # For the bump function of alpha-agents
hb     = torch.as_tensor(0.9) # For the bump function of beta-agents
varp   = torch.as_tensor(4.0) # Variance of the ZG initial distribution of agents
vMax   = torch.as_tensor(2.0) # Maximum velocity for the uniform initial distribution of agents' speed
vMin   = torch.as_tensor(-2.0) # Minimum velocity for the uniform initial distribution of agents' speed
c1     = torch.as_tensor(0.04) # Tracking gain 1
c2     = torch.sqrt(c1)  # Tracking gain 2
oMax   = torch.as_tensor(2.0) # Maximum radius obstacles
oMin   = torch.as_tensor(1.0) # Minimum radius obstacles
gap    = torch.as_tensor(5.0) # Minimum allowed distance between obstacles
static = True

parameters = {"na": na,
              "nb": nb,
              "d": d,
              "r": r,
              "e": e,
              "a": a,
              "b": b,
              "c": c,
              "ha": ha,
              "hb": hb,
              "varp": varp,
              "vMax": vMax,
              "vMin": vMin,
              "c1": c1,
              "c2": c2,
              "oMax": oMax,
              "oMin": oMin,
              "gap": gap,
              "static": static
              }

# Initialize the system to learn
real_system = realFlocking(parameters)
learned_system = torch.load('flocking_learn_system_phsa.pth')


q_obstac                         = generate_obstacles(nb, oMax, oMin, varp, gap)
q_dynamic, p_dynamic             = generate_leader(varp, nb, q_obstac, gap, static, vMax, vMin)
q_agents, p_agents               = generate_agents(na, varp, vMax, vMin, nb, q_obstac, gap)
inputs                           = torch.cat((q_obstac, q_agents, p_agents, q_dynamic, p_dynamic))
trajectory_real                  = real_system.sample(inputs, simulation_time, step_size)
trajectory_learned               = learned_system.sample(inputs, simulation_time, step_size)

laplacian_real    = real_system.laplacian(real_system.ha, trajectory_real[-1, 3*nb: 3*nb + 2*na])
laplacian_learned = learned_system.laplacian(learned_system.ha, trajectory_learned[-1, 3*nb: 3*nb + 2*na])

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots()
for i in range(na):
    for j in range(i+1, na):
        if laplacian_real[i, j] != 0:
            ax.plot([trajectory_real[-1, 3*nb + 2*i], trajectory_real[-1, 3*nb + 2*j]], [trajectory_real[-1, 3*nb + 2*i+1], trajectory_real[-1, 3*nb + 2*j+1]], 'b', linewidth=3)
for i in range(na):
    ax.plot(trajectory_real[:, 3*nb + 2*i], trajectory_real[:, 3*nb + 2*i+1], 'k')
    ax.plot(trajectory_real[0, 3*nb + 2*i], trajectory_real[0, 3*nb + 2*i+1], 'go', markersize=14)
    ax.plot(trajectory_real[-1, 3*nb + 2*i], trajectory_real[-1, 3*nb + 2*i+1], 'bs', markersize=14)
for i in range(nb):
    circle = plt.Circle((q_obstac[3*i+1], q_obstac[3*i+2]), q_obstac[3*i+0], color='k', linewidth=3)
    ax.add_patch(circle)
ax.plot(trajectory_real[0, 3*nb + 4*na], trajectory_real[0, 3*nb + 4*na + 1], 'rx', markersize=24)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.show()

fig, ax = plt.subplots()
for i in range(na):
    for j in range(i+1, na):
        if laplacian_learned[i, j] != 0:
            ax.plot([trajectory_learned[-1, 3*nb + 2*i], trajectory_learned[-1, 3*nb + 2*j]], [trajectory_learned[-1, 3*nb + 2*i+1], trajectory_learned[-1, 3*nb + 2*j+1]], 'b', linewidth=3)
for i in range(na):
    ax.plot(trajectory_learned[:, 3*nb + 2*i], trajectory_learned[:, 3*nb + 2*i+1], 'k')
    ax.plot(trajectory_learned[0, 3*nb + 2*i], trajectory_learned[0, 3*nb + 2*i+1], 'go', markersize=14)
    ax.plot(trajectory_learned[-1, 3*nb + 2*i], trajectory_learned[-1, 3*nb + 2*i+1], 'bs', markersize=14)
for i in range(nb):
    circle = plt.Circle((q_obstac[3*i+1], q_obstac[3*i+2]), q_obstac[3*i+0], color='k', linewidth=3)
    ax.add_patch(circle)
ax.plot(trajectory_learned[0, 3*nb + 4*na], trajectory_learned[0, 3*nb + 4*na + 1], 'rx', markersize=24)
plt.xlabel('x $[$m$]$')
plt.ylabel('y $[$m$]$')
plt.show()

epsilon_c_learned = (trajectory_learned[-1, 3*nb:3*nb + 4 * na] - trajectory_learned[-1, 3*nb + 4 * na:]).norm(p=2)

errors    = (trajectory_learned[:, 3*nb:3*nb + 4 * na] - trajectory_learned[:, 3*nb + 4 * na:]).norm(p=2, dim=1).le(0.05)
appear    = torch.where(errors == True)
epsilon_t_learned = appear[0]*step_size

epsilon_c_real = (trajectory_real[-1, 3*nb:3*nb + 4 * na] - trajectory_real[-1, 3*nb + 4 * na:]).norm(p=2)

errors    = (trajectory_real[:, 3*nb:3*nb + 4 * na] - trajectory_real[:, 3*nb + 4 * na:]).norm(p=2, dim=1).le(0.05)
appear    = torch.where(errors == True)
epsilon_t_real = appear[0]*step_size

epsilon_d = torch.sum((trajectory_real[:, 3*nb:3*nb + 4 * na] - trajectory_learned[:, 3*nb:3*nb + 4 * na]).norm(p=2, dim=1))

print('-------------------------------------------------')
print("Metrics to assess the difference between the trajectories from the ground-truth controller and the learned controller")
print('epsilon_c_learned = %.12f' % (epsilon_c_learned.detach().cpu().numpy()))
print('epsilon_c_real = %.12f' % (epsilon_c_real.detach().cpu().numpy()))
print('epsilon_t_learned = %.12f' % (epsilon_t_learned.detach().cpu().numpy()))
print('epsilon_t_real = %.12f' % (epsilon_t_real.detach().cpu().numpy()))
print('epsilon_d = %.12f' % (epsilon_d.detach().cpu().numpy()))
print('-------------------------------------------------')
