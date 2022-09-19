import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
from torch.autograd import Variable
from torch import nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Att_R(nn.Module):
    '''
      Attention for R
    '''

    def __init__(self, r, d, h, device):
        super().__init__()

        self.device = device
        self.activation_tanh = nn.Tanh()
        self.activation_soft = nn.Softmax(dim=2)
        self.r = r
        self.d = d
        self.Aq = nn.Parameter(torch.randn(r, d))
        self.Ak = nn.Parameter(torch.randn(r, d))
        self.Av = nn.Parameter(torch.randn(r, d))
        self.Ao = nn.Parameter(torch.randn(h, r))

    def forward(self, x, L):
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        Q = self.activation_tanh(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        K = self.activation_tanh(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_tanh(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))

        o = torch.bmm(self.activation_soft(torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(numNeighbors)).to(torch.float32), V)
        R = self.activation_tanh(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))
        R = R.reshape(int(x.shape[0] / x.shape[2]), x.shape[2], -1)
        R = R ** 2
        Rupper = torch.zeros([int(x.shape[0] / x.shape[2]), 4 * na, 4 * na], device=self.device)
        Rupper[:, :2 * na:2, :2 * na:2] = R[:, :, ::16]
        Rupper[:, :2 * na:2, 1:2 * na:2] = R[:, :, 1::16]
        Rupper[:, 1:2 * na:2, :2 * na:2] = R[:, :, 2::16]
        Rupper[:, 1:2 * na:2, 1:2 * na:2] = R[:, :, 3::16]
        Rupper[:, :2 * na:2, 2 * na::2] = R[:, :, 4::16]
        Rupper[:, :2 * na:2, 2 * na + 1::2] = R[:, :, 5::16]
        Rupper[:, 1:2 * na:2, 2 * na::2] = R[:, :, 6::16]
        Rupper[:, 1:2 * na:2, 2 * na + 1::2] = R[:, :, 7::16]
        Rupper[:, 2 * na::2, :2 * na:2] = R[:, :, 8::16]
        Rupper[:, 2 * na::2, 1:2 * na:2] = R[:, :, 9::16]
        Rupper[:, 2 * na + 1::2, :2 * na:2] = R[:, :, 10::16]
        Rupper[:, 2 * na + 1::2, 1:2 * na:2] = R[:, :, 11::16]
        Rupper[:, 2 * na::2, 2 * na::2] = R[:, :, 12::16]
        Rupper[:, 2 * na::2, 2 * na + 1::2] = R[:, :, 13::16]
        Rupper[:, 2 * na + 1::2, 2 * na::2] = R[:, :, 14::16]
        Rupper[:, 2 * na + 1::2, 2 * na + 1::2] = R[:, :, 15::16]
        Rupper = Rupper + torch.transpose(Rupper, 1, 2)
        Rdiag = torch.clone(Rupper)
        Rdiag[:, range(4 * na), range(4 * na)] = torch.zeros(4 * na, device=self.device)
        Rout = torch.eye(4 * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper
        R = Rout + torch.eye(4 * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R


class Att_J(nn.Module):
    '''
      Attention for J
    '''

    def __init__(self, r, d, h, device):
        super().__init__()

        self.device = device
        self.activation_tanh = nn.Tanh()
        self.activation_soft = nn.Softmax(dim=2)
        self.r = r
        self.d = d
        self.Aq = nn.Parameter(torch.randn(r, d))
        self.Ak = nn.Parameter(torch.randn(r, d))
        self.Av = nn.Parameter(torch.randn(r, d))
        self.Ao = nn.Parameter(torch.randn(h, r))

    def forward(self, x, L):
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        Q = self.activation_tanh(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        K = self.activation_tanh(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_tanh(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))

        o = torch.bmm(self.activation_soft(torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(numNeighbors)).to(torch.float32), V)
        J = self.activation_tanh(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o)).squeeze(dim=1)
        J = J.reshape(int(x.shape[0] / x.shape[2]), x.shape[2], -1)
        J = J ** 2
        Jupper = torch.zeros([int(x.shape[0] / x.shape[2]), 4 * na, 4 * na], device=self.device)
        Jupper[:, :2 * na:2, :2 * na:2] = J[:, :, ::16]
        Jupper[:, :2 * na:2, 1:2 * na:2] = J[:, :, 1::16]
        Jupper[:, 1:2 * na:2, :2 * na:2] = J[:, :, 2::16]
        Jupper[:, 1:2 * na:2, 1:2 * na:2] = J[:, :, 3::16]
        Jupper[:, :2 * na:2, 2 * na::2] = J[:, :, 4::16]
        Jupper[:, :2 * na:2, 2 * na + 1::2] = J[:, :, 5::16]
        Jupper[:, 1:2 * na:2, 2 * na::2] = J[:, :, 6::16]
        Jupper[:, 1:2 * na:2, 2 * na + 1::2] = J[:, :, 7::16]
        Jupper[:, 2 * na::2, :2 * na:2] = J[:, :, 8::16]
        Jupper[:, 2 * na::2, 1:2 * na:2] = J[:, :, 9::16]
        Jupper[:, 2 * na + 1::2, :2 * na:2] = J[:, :, 10::16]
        Jupper[:, 2 * na + 1::2, 1:2 * na:2] = J[:, :, 11::16]
        Jupper[:, 2 * na::2, 2 * na::2] = J[:, :, 12::16]
        Jupper[:, 2 * na::2, 2 * na + 1::2] = J[:, :, 13::16]
        Jupper[:, 2 * na + 1::2, 2 * na::2] = J[:, :, 14::16]
        Jupper[:, 2 * na + 1::2, 2 * na + 1::2] = J[:, :, 15::16]
        J = Jupper - torch.transpose(Jupper, 1, 2)

        return J


class Att_H(nn.Module):
    '''
      Attention for H
    '''

    def __init__(self, r, d, device):
        super().__init__()

        self.device = device
        self.activation_tanh = nn.Tanh()
        self.activation_swish = nn.SiLU()
        self.activation_soft = nn.Softmax(dim=2)
        self.r = r
        self.d = d
        self.Aq = nn.Parameter(torch.randn(r, d))
        self.Ak = nn.Parameter(torch.randn(r, d))
        self.Av = nn.Parameter(torch.randn(r, d))
        self.Ao = nn.Parameter(torch.randn(1, r))

    def forward(self, x, L):
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        Q = self.activation_tanh(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        K = self.activation_tanh(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_tanh(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))

        o = torch.bmm(self.activation_soft(torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(numNeighbors)).to(torch.float32), V)
        H = self.activation_swish(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o)).squeeze(dim=1)

        return H.sum(dim=1).pow(2).unsqueeze(dim=1)


class learnNavigation(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device = parameters["device"]
        self.na = parameters['na']
        self.R = Att_R(16, 4, 16, device=self.device).to(self.device)
        self.J = Att_J(16, 4, 16, device=self.device).to(self.device)
        self.H = Att_H(16, 4, device=self.device).to(self.device)

    def laplacian(self):
        L = torch.eye(self.na, device=self.device) - torch.diag(torch.ones(self.na - 1, device=self.device), diagonal=1) - torch.diag(torch.ones(self.na - 1, device=self.device), diagonal=-1)
        L[0, -1] = -1.0
        L[-1, 0] = -1.0
        return L

    def flocking_dynamics(self, t, inputs):
        LL = self.laplacian()
        L = torch.kron(LL, torch.ones((1, 4), device=self.device)).reshape(-1, 4 * self.na).repeat(inputs.shape[0], 1)
        inputsL = inputs[:, :4 * self.na] - inputs[:, 4 * self.na:]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, ::4] = pos[:, ::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, ::2]
        state[:, 3::4] = vel[:, 1::2]
        state = L * state

        R = self.R.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        J = self.J.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        inputsL = Variable(state.reshape(-1, self.na, 4).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dHdx = Hgrad[0]
        dHdx = dHdx.sum(dim=2).reshape(-1, 4 * self.na)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), (dHdx.to(torch.float32)).unsqueeze(dim=2)).squeeze(dim=2)

        return dx[:, :2 * self.na], dx[:, 2 * self.na:]

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros([inputs.shape[0], 2 * self.na], device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

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
        L        = torch.eye(self.na) + torch.diag(torch.ones(self.na-1), diagonal=1) + torch.diag(torch.ones(self.na-1), diagonal=-1)
        L[0, -1] = 1.0
        L[-1, 0] = 1.0
        return L

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
real_system       = realNavigation(parameters)
learned_system    = torch.load('navigation_learn_system_phsa.pth')
learned_system.na = na
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

q_agents, p_agents               = generate_agents()
q_dynamic, p_dynamic             = generate_leader()
inputs                           = torch.cat((q_agents, p_agents, q_dynamic, p_dynamic))
trajectory_real                  = real_system.sample(inputs, simulation_time, step_size)
trajectory_learned               = learned_system.forward(inputs.unsqueeze(dim=0).to(device), simulation_time, step_size).squeeze(dim=1).detach().cpu().numpy()

colors = get_cmap(na)
plt.rcParams.update({'font.size': 20})
plt.figure()
for i in range(na):
    plt.plot(trajectory_real[0, 4*na + 2*i], trajectory_real[0, 4*na + 2*i+1], color=colors(i), marker='x', markersize=28)
    plt.plot(trajectory_real[:, 2*i], trajectory_real[:, 2*i+1], color=colors(i), linewidth=2)
    plt.plot(trajectory_real[0, 2*i], trajectory_real[0, 2*i+1], color=colors(i), marker='o', markersize=12)
    plt.plot(trajectory_real[-1, 2*i], trajectory_real[-1, 2*i+1], color=colors(i), marker='s', markersize=12)
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