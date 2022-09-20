import torch
import matplotlib.pyplot as plt
from torch import nn
from torchdiffeq import odeint
from torch.autograd import Variable
import ctypes
import gc

global libc
libc = ctypes.CDLL("libc.so.6")

class realSystem():

    def __init__(self, parameters):

        self.na     = parameters['na']
        self.e      = parameters['e']
        self.d      = parameters['d']
        self.r      = parameters['r']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
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
                        grad_V[2 * i:2 * i + 2] -= z_sigma / self.sigma_norm(z_sigma)
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

def L2_loss(u, v):
    return (u - v).pow(2).mean()

class Att_R(nn.Module):
    '''
      Attention for R
    '''

    def __init__(self, r, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_tanh  = nn.Tanh()
        self.activation_soft  = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.r                = r
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.Aq               = nn.Parameter(torch.randn(r, d))
        self.Ak               = nn.Parameter(torch.randn(r, d))
        self.Av               = nn.Parameter(torch.randn(r, d))
        self.Aq1              = nn.Parameter(torch.randn(r, r))
        self.Ak1              = nn.Parameter(torch.randn(r, r))
        self.Av1              = nn.Parameter(torch.randn(r, r))
        self.Aq5              = nn.Parameter(torch.randn(r, r))
        self.Ak5              = nn.Parameter(torch.randn(r, r))
        self.Av5              = nn.Parameter(torch.randn(r, r))

        self.Ao  = nn.Parameter(torch.randn(r, r))
        self.Ao1 = nn.Parameter(torch.randn(r, r))
        self.Ao5 = nn.Parameter(torch.randn(h, r))

    def forward(self, x, L):

        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        # Self-Attention layers
        Q = self.activation_sigmo(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_sigmo(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        R = self.activation_swish(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R))
        V = self.activation_sigmo(torch.bmm(self.Av1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        R = self.activation_swish(torch.bmm(self.Ao1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R))
        V = self.activation_sigmo(torch.bmm(self.Av5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), R)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        R = self.activation_swish(torch.bmm(self.Ao5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        # Reshape and kronecker before post-processing
        del Q, K, V, o
        R11 = R[:, 0:4,   :].sum(1)
        R12 = R[:, 4:8,   :].sum(1)
        R21 = R[:, 8:12,  :].sum(1)
        R22 = R[:, 12:16, :].sum(1)
        R11 = R11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R12 = R12.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R21 = R21.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R22 = R22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        del R11, R12, R21, R22
        R   = R ** 2

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = int(torch.sqrt(torch.as_tensor(self.h)))
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

    def _assess(self, x, L):
        l  = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        R  = torch.zeros(x.shape[0], l * l, na,  device=self.device)

        # Self-Attention layers
        for i in range(na):
            Q = self.activation_sigmo(self.Aq @ x[i, :, :]).transpose(0, 1)
            K = self.activation_sigmo(self.Ak @ x[i, :, :])
            V = self.activation_sigmo(self.Av @ x[i, :, :]).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            r = self.activation_swish(self.Ao @ o)

            Q = self.activation_sigmo(self.Aq1 @ r).transpose(0, 1)
            K = self.activation_sigmo(self.Ak1 @ r)
            V = self.activation_sigmo(self.Av1 @ r).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            r = self.activation_swish(self.Ao1 @ o)

            Q = self.activation_sigmo(self.Aq5 @ r).transpose(0, 1)
            K = self.activation_sigmo(self.Ak5 @ r)
            V = self.activation_sigmo(self.Av5 @ r).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            R[i, :, :] = self.activation_swish(self.Ao5 @ o)

        # Reshape and kronecker before post-processing
        R11 = R[:, 0:4,   :].sum(1)
        R12 = R[:, 4:8,   :].sum(1)
        R21 = R[:, 8:12,  :].sum(1)
        R22 = R[:, 12:16, :].sum(1)
        R11 = R11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R12 = R12.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R21 = R21.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R22 = R22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        del o, Q, K, V
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        del R22, R21, R12, R11
        R   = R ** 2

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = int(torch.sqrt(torch.as_tensor(self.h)))
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        del Rupper, Rdiag

        return R

class Att_J(nn.Module):
    '''
      Attention for J
    '''

    def __init__(self, r, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_tanh  = nn.Tanh()
        self.activation_soft  = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_sigmo = nn.Sigmoid()
        self.r                = r
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.Aq               = nn.Parameter(torch.randn(r, d))
        self.Ak               = nn.Parameter(torch.randn(r, d))
        self.Av               = nn.Parameter(torch.randn(r, d))
        self.Aq1              = nn.Parameter(torch.randn(r, r))
        self.Ak1              = nn.Parameter(torch.randn(r, r))
        self.Av1              = nn.Parameter(torch.randn(r, r))
        self.Aq5              = nn.Parameter(torch.randn(r, r))
        self.Ak5              = nn.Parameter(torch.randn(r, r))
        self.Av5              = nn.Parameter(torch.randn(r, r))

        self.Ao  = nn.Parameter(torch.randn(r, r))
        self.Ao1 = nn.Parameter(torch.randn(r, r))
        self.Ao5 = nn.Parameter(torch.randn(h, r))

    def forward(self, x, L):

        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        # Self-Attention layers
        Q = self.activation_sigmo(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_sigmo(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        J = self.activation_swish(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J))
        V = self.activation_sigmo(torch.bmm(self.Av1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        J = self.activation_swish(torch.bmm(self.Ao1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J))
        V = self.activation_sigmo(torch.bmm(self.Av5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), J)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        J = self.activation_swish(torch.bmm(self.Ao5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        # Reshape, kronecker and post-processing to ensure skew-symmetry
        del Q, K, V, o
        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        J11 = J11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J11, J12, J21, J22, j12, j21

        return J

    def _assess(self, x, L):
        l  = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        J  = torch.zeros(x.shape[0], l * l, na,  device=self.device)

        # Self-Attention layers
        for i in range(na):
            Q = self.activation_sigmo(self.Aq @ x[i, :, :]).transpose(0, 1)
            K = self.activation_sigmo(self.Ak @ x[i, :, :])
            V = self.activation_sigmo(self.Av @ x[i, :, :]).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            j = self.activation_swish(self.Ao @ o)

            Q = self.activation_sigmo(self.Aq1 @ j).transpose(0, 1)
            K = self.activation_sigmo(self.Ak1 @ j)
            V = self.activation_sigmo(self.Av1 @ j).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            j = self.activation_swish(self.Ao1 @ o)

            Q = self.activation_sigmo(self.Aq5 @ j).transpose(0, 1)
            K = self.activation_sigmo(self.Ak5 @ j)
            V = self.activation_sigmo(self.Av5 @ j).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            J[i, :, :] = self.activation_swish(self.Ao5 @ o)

        # Reshape, kronecker and post-processing to ensure skew-symmetry
        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        J11 = J11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))
        del o, Q, K, V
        J   = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J22, J21, J12, J11

        return J

class Att_H(nn.Module):
    '''
      Attention for H
    '''

    def __init__(self, r, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_tanh  = nn.Tanh()
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_soft  = nn.Softmax(dim=2)
        self.activation_softA = nn.Softmax(dim=1)
        self.r = r
        self.d = d
        self.h = h

        # Initialized to avoid unstable training
        self.Aq               = nn.Parameter(torch.randn(r, d))
        self.Ak               = nn.Parameter(torch.randn(r, d))
        self.Av               = nn.Parameter(torch.randn(r, d))
        self.Aq1              = nn.Parameter(torch.randn(r, r))
        self.Ak1              = nn.Parameter(torch.randn(r, r))
        self.Av1              = nn.Parameter(torch.randn(r, r))
        self.Aq5              = nn.Parameter(torch.randn(r, r))
        self.Ak5              = nn.Parameter(torch.randn(r, r))
        self.Av5              = nn.Parameter(torch.randn(r, r))

        self.Ao  = nn.Parameter(torch.randn(r, r))
        self.Ao1 = nn.Parameter(torch.randn(r, r))
        self.Ao5 = nn.Parameter(torch.randn(h, r))

    def forward(self, x, L):
        l = 2
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        # Self-Attention layers
        Q = self.activation_sigmo(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_sigmo(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        M = self.activation_swish(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M))
        V = self.activation_sigmo(torch.bmm(self.Av1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        M = self.activation_swish(torch.bmm(self.Ao1.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        Q = self.activation_sigmo(torch.bmm(self.Aq5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M))
        V = self.activation_sigmo(torch.bmm(self.Av5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), M)).transpose(1, 2)

        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        M = self.activation_swish(torch.bmm(self.Ao5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        # Reshape, kronecker and post-processing
        del Q, K, V, o
        M11 = torch.kron((M[:, :int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, int(self.d / na):2 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 2 * int(self.d / na):3 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 3 * int(self.d / na):4 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 4 * int(self.d / na):, :] ** 2).sum(1)

        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        del Mupper11, Mupper12, Mupper21, Mupper22
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        q = x[:, :4, :].transpose(1, 2).reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

    def _assess(self, x, L):
        l = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        M  = torch.zeros(x.shape[0], l * l, na,  device=self.device)

        # Self-Attention layers
        for i in range(na):
            Q = self.activation_sigmo(self.Aq @ x[i, :, :]).transpose(0, 1)
            K = self.activation_sigmo(self.Ak @ x[i, :, :])
            V = self.activation_sigmo(self.Av @ x[i, :, :]).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            m = self.activation_swish(self.Ao @ o)

            Q = self.activation_sigmo(self.Aq1 @ m).transpose(0, 1)
            K = self.activation_sigmo(self.Ak1 @ m)
            V = self.activation_sigmo(self.Av1 @ m).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            m = self.activation_swish(self.Ao1 @ o)

            Q = self.activation_sigmo(self.Aq5 @ m).transpose(0, 1)
            K = self.activation_sigmo(self.Ak5 @ m)
            V = self.activation_sigmo(self.Av5 @ m).transpose(0, 1)

            o = (self.activation_softA(Q @ K / torch.sqrt(numNeighbors[i])).to(torch.float32) @ V).transpose(0, 1)
            M[i, :, :] = self.activation_swish(self.Ao5 @ o)

        del o, Q, K, V

        # Reshape, kronecker and post-processing
        M11 = torch.kron((M[:, :int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, int(self.d / na):2 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 2 * int(self.d / na):3 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 3 * int(self.d / na):4 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 4 * int(self.d / na):, :] ** 2).sum(1)
        l = int(torch.sqrt(torch.as_tensor(self.h))/2)

        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        q = x[:, :4, :].transpose(1, 2).reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

class learnSystem(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = Att_R(8, 4, 4 * 4, device=self.device).to(self.device)
        self.J       = Att_J(8, 4, 1, device=self.device).to(self.device)
        self.H       = Att_H(8, 6, 5 * 5, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def navigation_dynamics(self, t, inputs):

        # Obtain Laplacians
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)

        # Get inputs for the self-attention modules
        inputs_l         = inputs[:, :4 * self.na]
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2]
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2]
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2]
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2]

        # Obtain relative states
        pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1)
        vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1)

        # Input for R and J
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = state_d
        state                                        = state.reshape(-1, 4 * self.na)

        # Mask out non-neighboring robots
        state = L * state

        # R and J
        R = self.R.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        J = self.J.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        # Input for H
        state_h = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        state_h[:, 0::6] = torch.kron(state_d[:, :, 0], torch.ones(self.na, 1, device=self.device))
        state_h[:, 1::6] = torch.kron(state_d[:, :, 1], torch.ones(self.na, 1, device=self.device))
        state_h[:, 2::6] = torch.kron(state_d[:, :, 2], torch.ones(self.na, 1, device=self.device))
        state_h[:, 3::6] = torch.kron(state_d[:, :, 3], torch.ones(self.na, 1, device=self.device))
        state_h[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        state_h[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6]   = Q
        LLL               = LLL.reshape(-1, 6 * self.na)
        state_h           = LLL * state_h

        # H
        inputs_l = Variable(state_h.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputs_l.to(torch.float32), LLL.reshape(-1, self.na, 6).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        del state_d, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputs_l, J, R
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd     = self.leader_dynamics(t, inputs)
        da     = self.navigation_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

    def navigation_dynamics_assess(self, inputs):

        # Obtain Laplacians
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)

        # Get inputs for the self-attention modules
        inputs_l         = inputs[:, :4 * self.na]
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2]
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2]
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2]
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2]

        # Obtain relative states
        pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1)
        vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1)

        # Input for R and J
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = state_d
        state                                        = state.reshape(-1, 4 * self.na)

        # Mask out non-neighboring robots
        state = L * state

        # R and J
        R = self.R._assess(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        J = self.J._assess(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        # Input for H
        state_h = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        state_h[:, 0::6] = torch.kron(state_d[:, :, 0], torch.ones(self.na, 1, device=self.device))
        state_h[:, 1::6] = torch.kron(state_d[:, :, 1], torch.ones(self.na, 1, device=self.device))
        state_h[:, 2::6] = torch.kron(state_d[:, :, 2], torch.ones(self.na, 1, device=self.device))
        state_h[:, 3::6] = torch.kron(state_d[:, :, 3], torch.ones(self.na, 1, device=self.device))
        state_h[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        state_h[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6]   = Q
        LLL               = LLL.reshape(-1, 6 * self.na)
        state_h           = LLL * state_h

        # H
        inputs_l = Variable(state_h.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        H = self.H._assess(inputs_l.to(torch.float32), LLL.reshape(-1, self.na, 6).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        del state_d, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputs_l, J, R

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics_assess(self, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics_assess(self, inputs):
        dd = self.leader_dynamics_assess(inputs)
        da = self.navigation_dynamics_assess(inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def assess(self, inputs, simulation_time, step_size):
        outputs_2 = torch.zeros(len(simulation_time), inputs.shape[1], device=self.device)
        for i in range(len(simulation_time)):
            outputs_2[i, :] = inputs + step_size*self.overall_dynamics_assess(inputs)
            inputs          = torch.clone(outputs_2[i, :].unsqueeze(0))
            # Free unused memory
            if i % 5 == 0:
                gc.collect()
                libc.malloc_trim()
        return outputs_2

class MLP_R(nn.Module):
    '''
      MLP for R
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A                = nn.Parameter(0.1*torch.randn(d, h))

    def forward(self, x):
        # MLP layer
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape and kronecker before post-processing
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        R  = R.reshape(x.shape[0], l * l, na)

        R11 = R[:, 0:4,   :].sum(1)
        R12 = R[:, 4:8,   :].sum(1)
        R21 = R[:, 8:12,  :].sum(1)
        R22 = R[:, 12:16, :].sum(1)
        R11 = R11.reshape(int(x.shape[0] / na), na, na)
        R12 = R12.reshape(int(x.shape[0] / na), na, na)
        R21 = R21.reshape(int(x.shape[0] / na), na, na)
        R22 = R22.reshape(int(x.shape[0] / na), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        R   = R ** 2
        del R11, R12, R21, R22

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = 4
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

class MLP_J(nn.Module):
    '''
      MLP for J
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A                = nn.Parameter(0.1*torch.randn(d, h))

    def forward(self, x):
        # MLP layer
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing to ensure skew-symmetry
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        J  = J.reshape(x.shape[0], l * l, na)

        J11 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        J11 = J11.reshape(int(x.shape[0] / na), na, na)
        J12 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / na), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J11, J12, J21, J22

        return J

class MLP_H(nn.Module):
    '''
      MLP for H
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d = d
        self.h = h

        # Initialized to avoid unstable training
        self.A = nn.Parameter(0.1*torch.randn(d, h))

    def forward(self, x):
        # MLP layer
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, :int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, int(self.d / na):2 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 2 * int(self.d / na):3 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 3 * int(self.d / na):4 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 4 * int(self.d / na):, :] ** 2).sum(1)

        l = 2
        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        del Mupper11, Mupper12, Mupper21, Mupper22

        q = torch.zeros(x.shape[0], 4 * na, device=self.device)
        q[:, 0::4] = x[:, 0::6]
        q[:, 1::4] = x[:, 1::6]
        q[:, 2::4] = x[:, 2::6]
        q[:, 3::4] = x[:, 3::6]
        q          = q.reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

class learnSystemMLP(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = MLP_R(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.J       = MLP_J(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.H       = MLP_H(6 * self.na, 5 * 5 * self.na, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def navigation_dynamics(self, t, inputs):
        # Obtain Laplacians
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)

        # Get inputs for the MLP modules
        inputs_l         = inputs[:, :4 * self.na]
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2]
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2]
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2]
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2]

        # Obtain relative states
        pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1)
        vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1)

        # Input for R and J
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = state_d
        state                                        = state.reshape(-1, 4 * self.na)

        # Mask out non-neighboring robots
        state = L * state

        # R and J
        R = self.R.forward(state)

        J = self.J.forward(state)

        # Input for H
        state_h = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        state_h[:, 0::6] = torch.kron(state_d[:, :, 0], torch.ones(self.na, 1, device=self.device))
        state_h[:, 1::6] = torch.kron(state_d[:, :, 1], torch.ones(self.na, 1, device=self.device))
        state_h[:, 2::6] = torch.kron(state_d[:, :, 2], torch.ones(self.na, 1, device=self.device))
        state_h[:, 3::6] = torch.kron(state_d[:, :, 3], torch.ones(self.na, 1, device=self.device))
        state_h[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        state_h[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6] = Q
        LLL = LLL.reshape(-1, 6 * self.na)
        state_h = LLL * state_h

        del LL, inputs_d, inputs_l, inputs, pos, pos1, pos2, vel, vel1, vel2, state_d, Q, LLL

        # H
        inputs_l = Variable(state_h.data, requires_grad=True)
        H       = self.H.forward(inputs_l.to(torch.float32))
        Hgrad   = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        dH      = Hgrad[0].reshape(state.shape[0], self.na, 6)
        dHq     = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ     = dHq[:, range(self.na), range(self.na), :]
        dHp     = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP     = dHp[:, range(self.na), range(self.na), :]
        dHdx    = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd     = self.leader_dynamics(t, inputs)
        da     = self.navigation_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

class GNN_R(nn.Module):
    '''
      GNN for R
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A               = nn.Parameter(0.015*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape and kronecker before post-processing
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        R  = R.reshape(x.shape[0], l * l, na)

        R11 = R[:, 0:4,   :].sum(1)
        R12 = R[:, 4:8,   :].sum(1)
        R21 = R[:, 8:12,  :].sum(1)
        R22 = R[:, 12:16, :].sum(1)
        R11 = R11.reshape(int(x.shape[0] / na), na, na)
        R12 = R12.reshape(int(x.shape[0] / na), na, na)
        R21 = R21.reshape(int(x.shape[0] / na), na, na)
        R22 = R22.reshape(int(x.shape[0] / na), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        R   = R ** 2
        del R11, R12, R21, R22

        # Operations to ensure sparsity and positive semidefiniteness
        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = 4
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

class GNN_J(nn.Module):
    '''
      GNN for J
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A                = nn.Parameter(0.015*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing to ensure skew-symmetry
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        J  = J.reshape(x.shape[0], l * l, na)

        J11 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        J11 = J11.reshape(int(x.shape[0] / na), na, na)
        J12 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / na), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J11, J12, J21, J22

        return J

class GNN_H(nn.Module):
    '''
      MLP for H
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d = d
        self.h = h

        # Initialized to avoid unstable training
        self.A = nn.Parameter(0.015*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, :int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, int(self.d / na):2 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 2 * int(self.d / na):3 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 3 * int(self.d / na):4 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 4 * int(self.d / na):, :] ** 2).sum(1)

        l = 2
        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        del Mupper11, Mupper12, Mupper21, Mupper22

        q = torch.zeros(x.shape[0], 4 * na, device=self.device)
        q[:, 0::4] = x[:, 0::6]
        q[:, 1::4] = x[:, 1::6]
        q[:, 2::4] = x[:, 2::6]
        q[:, 3::4] = x[:, 3::6]
        q          = q.reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

class learnSystemGNN(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = GNN_R(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.J       = GNN_J(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.H       = GNN_H(6 * self.na, 5 * 5 * self.na, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def navigation_dynamics(self, t, inputs):
        # Obtain Laplacian
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)

        # Get inputs for the GNN modules
        inputs_l         = inputs[:, :4 * self.na]
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2]
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2]
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2]
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2]

        # Obtain relative states
        pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1)
        vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1)

        # Input for R and J
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = state_d
        state                                        = state.transpose(2, 3).transpose(1, 2)

        state_h = torch.zeros((inputs.shape[0], 4, self.na, self.na), device=self.device)
        state_h[:, 0, :, :] = torch.bmm(LL, state[:, 0, :, :])
        state_h[:, 1, :, :] = torch.bmm(LL, state[:, 1, :, :])
        state_h[:, 2, :, :] = torch.bmm(LL, state[:, 2, :, :])
        state_h[:, 3, :, :] = torch.bmm(LL, state[:, 3, :, :])
        state_h             = state_h.transpose(1, 2).transpose(2, 3).reshape(-1, 4 * self.na)

        # R and J
        R = self.R.forward(state_h)

        J = self.J.forward(state_h)

        # Input for H
        state_h = torch.zeros((state.shape[0] * self.na, 6 * self.na), device=self.device)
        state_h[:, 0::6] = torch.kron(state_d[:, :, 0], torch.ones(self.na, 1, device=self.device))
        state_h[:, 1::6] = torch.kron(state_d[:, :, 1], torch.ones(self.na, 1, device=self.device))
        state_h[:, 2::6] = torch.kron(state_d[:, :, 2], torch.ones(self.na, 1, device=self.device))
        state_h[:, 3::6] = torch.kron(state_d[:, :, 3], torch.ones(self.na, 1, device=self.device))
        state_h[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        state_h[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        state_h          = state_h.reshape(-1, self.na, self.na, 6)
        state_h          = state_h.transpose(2, 3).transpose(1, 2)
        state_h[:, 0, :, :] = torch.bmm(LL, state_h[:, 0, :, :])
        state_h[:, 1, :, :] = torch.bmm(LL, state_h[:, 1, :, :])
        state_h[:, 2, :, :] = torch.bmm(LL, state_h[:, 2, :, :])
        state_h[:, 3, :, :] = torch.bmm(LL, state_h[:, 3, :, :])
        state_h[:, 4, :, :] = torch.bmm(Q, state_h[:, 4, :, :])
        state_h[:, 5, :, :] = torch.bmm(LL, state_h[:, 5, :, :])
        state_h             = state_h.transpose(1, 2).transpose(2, 3).reshape(-1, 6 * self.na)

        del state, LL, inputs_d, inputs_l, inputs, pos, pos1, pos2, vel, vel1, vel2, state_d, Q

        # H
        inputs_l = Variable(state_h.data, requires_grad=True)
        H = self.H.forward(inputs_l.to(torch.float32))
        Hgrad = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        dH = Hgrad[0].reshape(state_h.shape[0], self.na, 6)
        dHq = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd     = self.leader_dynamics(t, inputs)
        da     = self.navigation_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

class Att_mask(nn.Module):
    '''
      Attention for L
    '''

    def __init__(self, r, d, device):
        super().__init__()

        self.device           = device
        self.activation_tanh  = nn.Tanh()
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.activation_soft  = nn.Softmax(dim=2)
        self.r                = r
        self.d                = d

        # Initialized to avoid unstable training
        self.Aq               = nn.Parameter(0.50*torch.randn(r, d))
        self.Ak               = nn.Parameter(0.50*torch.randn(r, d))
        self.Av               = nn.Parameter(0.50*torch.randn(r, d))
        self.Ao               = nn.Parameter(0.50*torch.randn(1, r))

    def forward(self, x, L):

        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        # Self-Attention layer
        Q = self.activation_tanh(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        K = self.activation_tanh(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_tanh(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))

        o = torch.bmm(self.activation_soft(torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(numNeighbors)).to(torch.float32), V)
        R = self.activation_tanh(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o)).squeeze(dim=1)
        del Q, K, V, o

        # Operations to ensure Laplacian-like structure
        R = R.reshape(int(x.shape[0]/x.shape[2]), x.shape[2], x.shape[2])
        R = R**2
        Rdiag  = torch.clone(R)
        Rdiag[:, range(x.shape[2]), range(x.shape[2])] = torch.zeros(x.shape[2], device=self.device)
        Rout   = torch.eye(x.shape[2], device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0]/x.shape[2]), 1, 1) * R
        R      = Rout + torch.eye(x.shape[2], device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0]/x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

class GNNSA_R(nn.Module):
    '''
      GNN for R
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A               = nn.Parameter(0.2*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape and kronecker before post-processing
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        R  = R.reshape(x.shape[0], l * l, na)

        R11 = R[:, 0:4,   :].sum(1)
        R12 = R[:, 4:8,   :].sum(1)
        R21 = R[:, 8:12,  :].sum(1)
        R22 = R[:, 12:16, :].sum(1)
        R11 = R11.reshape(int(x.shape[0] / na), na, na)
        R12 = R12.reshape(int(x.shape[0] / na), na, na)
        R21 = R21.reshape(int(x.shape[0] / na), na, na)
        R22 = R22.reshape(int(x.shape[0] / na), na, na)
        R11 = torch.kron(R11, torch.eye(2, device=self.device).unsqueeze(0))
        R12 = torch.kron(R12, torch.eye(2, device=self.device).unsqueeze(0))
        R21 = torch.kron(R21, torch.eye(2, device=self.device).unsqueeze(0))
        R22 = torch.kron(R22, torch.eye(2, device=self.device).unsqueeze(0))
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        R   = R ** 2
        del R11, R12, R21, R22

        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = 4
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / na), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

class GNNSA_J(nn.Module):
    '''
      GNN for J
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d                = d
        self.h                = h

        # Initialized to avoid unstable training
        self.A                = nn.Parameter(0.2*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing to ensure skew-symmetry
        l  = 4
        na = int(x.shape[1]/torch.as_tensor(4))
        J  = J.reshape(x.shape[0], l * l, na)

        J11 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / na), na)
        J11 = J11.reshape(int(x.shape[0] / na), na, na)
        J12 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J21 = torch.zeros((int(x.shape[0] / na), na, na), device=self.device)
        J22 = J22.reshape(int(x.shape[0] / na), na, na)
        J12[:, range(na), range(na)] = j12
        J21[:, range(na), range(na)] = j21
        J11 = torch.kron(J11, torch.eye(2, device=self.device).unsqueeze(0))
        J12 = torch.kron(J12, torch.eye(2, device=self.device).unsqueeze(0))
        J21 = torch.kron(J21, torch.eye(2, device=self.device).unsqueeze(0))
        J22 = torch.kron(J22, torch.eye(2, device=self.device).unsqueeze(0))
        J = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J11, J12, J21, J22

        return J

class GNNSA_H(nn.Module):
    '''
      MLP for H
    '''

    def __init__(self, d, h, device):
        super().__init__()

        self.device           = device
        self.activation_swish = nn.SiLU()
        self.d = d
        self.h = h

        # Initialized to avoid unstable training
        self.A = nn.Parameter(0.2*torch.randn(d, h))

    def forward(self, x):
        # GNN filter
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))

        # Reshape, kronecker and post processing
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, :int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, int(self.d / na):2 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 2 * int(self.d / na):3 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 3 * int(self.d / na):4 * int(self.d / na), :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 4 * int(self.d / na):, :] ** 2).sum(1)

        l = 2
        Mupper11 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper12 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper21 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)
        Mupper22 = torch.zeros([x.shape[0], l * na, l * na], device=self.device)

        Mupper11[:, range(l * na), range(l * na)] = M11
        Mupper12[:, range(l * na), range(l * na)] = M12
        Mupper21[:, range(l * na), range(l * na)] = M21
        Mupper22[:, range(l * na), range(l * na)] = M22

        del M11, M12, M21, M22

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        del Mupper11, Mupper12, Mupper21, Mupper22

        q = torch.zeros(x.shape[0], 4 * na, device=self.device)
        q[:, 0::4] = x[:, 0::6]
        q[:, 1::4] = x[:, 1::6]
        q[:, 2::4] = x[:, 2::6]
        q[:, 3::4] = x[:, 3::6]
        q          = q.reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

class learnSystemGNNSA(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = GNNSA_R(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.J       = GNNSA_J(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.H       = GNNSA_H(6 * self.na, 5 * 5 * self.na, device=self.device).to(self.device)
        self.mask    = Att_mask(8, 4, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def navigation_dynamics(self, t, inputs):
        # Obtain Laplacians
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)

        # Get inputs for the GNN and SA modules
        inputs_l         = inputs[:, :4 * self.na]
        inputs_d         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        state_d          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        state_d[:, :, 0] = inputs_d[:, 0:2 * self.na:2]
        state_d[:, :, 1] = inputs_d[:, 1:2 * self.na:2]
        state_d[:, :, 2] = inputs_d[:, 2 * self.na + 0::2]
        state_d[:, :, 3] = inputs_d[:, 2 * self.na + 1::2]

        # Obtain relative states
        pos1 = inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputs_l[:, :2 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputs_l.shape[0] * self.na, -1)
        vel1 = inputs_l[:, 2 * self.na:6 * self.na].reshape(inputs_l.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputs_l[:, 2 * self.na:4 * self.na].reshape(inputs_l.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputs_l.shape[0] * self.na, -1)

        # Input for R and J
        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = state_d
        state = state.transpose(2, 3).transpose(1, 2)

        # Mask Laplacian with the self-attention
        mask = self.mask.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        mask_l = mask * LL
        mask_q = mask * Q

        state_h = torch.zeros((inputs.shape[0], 4, self.na, self.na), device=self.device)
        state_h[:, 0, :, :] = torch.bmm(mask_l, state[:, 0, :, :])
        state_h[:, 1, :, :] = torch.bmm(mask_l, state[:, 1, :, :])
        state_h[:, 2, :, :] = torch.bmm(mask_l, state[:, 2, :, :])
        state_h[:, 3, :, :] = torch.bmm(mask_l, state[:, 3, :, :])
        state_h = state_h.transpose(1, 2).transpose(2, 3).reshape(-1, 4 * self.na)

        # R and J
        R = self.R.forward(state_h)

        J = self.J.forward(state_h)

        # Input for H
        state_h = torch.zeros((state_h.shape[0], 6 * self.na), device=self.device)
        state_h[:, 0::6] = torch.kron(state_d[:, :, 0], torch.ones(self.na, 1, device=self.device))
        state_h[:, 1::6] = torch.kron(state_d[:, :, 1], torch.ones(self.na, 1, device=self.device))
        state_h[:, 2::6] = torch.kron(state_d[:, :, 2], torch.ones(self.na, 1, device=self.device))
        state_h[:, 3::6] = torch.kron(state_d[:, :, 3], torch.ones(self.na, 1, device=self.device))
        state_h[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        state_h[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        state_h = state_h.reshape(-1, self.na, self.na, 6)
        state_h = state_h.transpose(2, 3).transpose(1, 2)
        state_h[:, 0, :, :] = torch.bmm(mask_l, state_h[:, 0, :, :])
        state_h[:, 1, :, :] = torch.bmm(mask_l, state_h[:, 1, :, :])
        state_h[:, 2, :, :] = torch.bmm(mask_l, state_h[:, 2, :, :])
        state_h[:, 3, :, :] = torch.bmm(mask_l, state_h[:, 3, :, :])
        state_h[:, 4, :, :] = torch.bmm(mask_q, state_h[:, 4, :, :])
        state_h[:, 5, :, :] = torch.bmm(mask_l, state_h[:, 5, :, :])
        state_h = state_h.transpose(1, 2).transpose(2, 3).reshape(-1, 6 * self.na)

        del state, LL, inputs_d, inputs_l, inputs, pos, pos1, pos2, vel, vel1, vel2, state_d, Q

        # H
        inputs_l = Variable(state_h.data, requires_grad=True)
        H = self.H.forward(inputs_l.to(torch.float32))
        Hgrad = torch.autograd.grad(H.sum(), inputs_l, only_inputs=True, create_graph=True)
        dH = Hgrad[0].reshape(state_h.shape[0], self.na, 6)
        dHq = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        # Closed-loop dynamics
        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd     = self.leader_dynamics(t, inputs)
        da     = self.navigation_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

def generate_agents(na):
    q_agents       = torch.zeros(2 * na)
    q_agents[0::2] = torch.cat((3.0 * torch.ones(int(na/2)), -3.0 * torch.ones(int(na/2))))
    q_agents[1::2] = torch.cat((torch.linspace(-3.0 * (int(na/4)-1) - 1.5, 3.0 * (int(na/4)-1) + 1.5, int(na/2)),
                                torch.linspace(3.0 * (int(na/4)-1) + 1.5, -3.0 * (int(na/4)-1) - 1.5, int(na/2))))

    return q_agents + 1.0 * torch.rand(2 * na), torch.zeros(2 * na)

def generate_leader(na):
    q_agents       = torch.zeros(2 * na)
    q_agents[0::2] = torch.cat((-3.0 * torch.ones(int(na/2)), 3.0 * torch.ones(int(na/2))))
    q_agents[1::2] = torch.cat((torch.linspace(3.0 * (int(na/4)-1) + 1.5, -3.0 * (int(na/4)-1) - 1.5, int(na/2)),
                                torch.linspace(-3.0 * (int(na/4)-1) - 1.5, 3.0 * (int(na/4)-1) + 1.5, int(na/2))))

    return q_agents + 1.0 * torch.rand(2 * na), torch.zeros(2 * na)

def get_cmap(n):
    return plt.cm.get_cmap('hsv', n+1)