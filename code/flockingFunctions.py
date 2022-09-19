import torch
import matplotlib.pyplot as plt
from torch import nn
from torchdiffeq import odeint
from torch.autograd import Variable
import ctypes
import gc

global libc
libc = ctypes.CDLL("libc.so.6")


class realFlocking():

    def __init__(self, parameters):

        self.na     = parameters['na']
        self.d      = parameters['d']
        self.r      = parameters['r']
        self.e      = parameters['e']
        self.a      = parameters['a']
        self.b      = parameters['b']
        self.c      = parameters['c']
        self.ha     = parameters['ha']
        self.c1     = parameters['c1']
        self.c2     = parameters['c2']

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
        return -self.c1 * (q_agents - q_dynamic) - self.c2 * (p_agents - p_dynamic)

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

    def flocking_dynamics(self, t, inputs):

        L         = self.augmented_laplacian(self.ha, inputs[:2 * self.na])
        DeltaV    = self.grad_V(self.ha, L, inputs[:2 * self.na])

        dq        = inputs[2 * self.na:4 * self.na]
        dp        = -DeltaV \
                    -L @ inputs[2 * self.na:4 * self.na] \
                    +self.f_control(inputs[:2 * self.na],
                                    inputs[2 * self.na:4 * self.na],
                                    inputs[4 * self.na:6 * self.na],
                                    inputs[6 * self.na:])

        return dq, dp

    def leader_dynamics(self, t, inputs):
        return inputs[6 * self.na:], torch.zeros(2 * self.na)

    def overall_dynamics(self, t, inputs):
        da = self.flocking_dynamics(t, inputs)
        dd = self.leader_dynamics(t, inputs)
        return torch.cat((da[0], da[1], dd[0], dd[1]))

    def sample(self, inputs, simulation_time, step_size):
        targets = odeint(self.overall_dynamics, inputs, simulation_time, method='euler', options={'step_size': step_size})
        return targets


# def L2_loss(u, v, dH, device):
#     na = int(u.shape[1]/4)
#     tu1 = torch.kron(u.reshape(-1, na, 2), torch.ones(1, na, 1, device=device))
#     tu2 = u.reshape(-1, na, 2).repeat(1, na, 1)
#     tv1 = torch.kron(v.reshape(-1, na, 2), torch.ones(1, na, 1, device=device))
#     tv2 = v.reshape(-1, na, 2).repeat(1, na, 1)
#     du  = (tu1 - tu2).norm(p=2, dim=2)
#     dv  = (tv1 - tv2).norm(p=2, dim=2)
#     du  = torch.mean(torch.kthvalue(du, na + 1, 1, True)[0])
#     dv  = torch.mean(torch.kthvalue(dv, na + 1, 1, True)[0])
#     return (u - v).pow(2).mean() + 1 / ((u[:, 2*na:]).pow(2).mean() + 1e-5) * 2 + dH.pow(2).mean() * 1000 #+ torch.abs(du-dv) * 0.5

def L2_loss(u, v):
    return (u - v).pow(2).mean()

def LD_loss(u, v, device):
    na = int(u.shape[1]/4)
    tu1 = torch.kron(u.reshape(-1, na, 2), torch.ones(1, na, 1, device=device))
    tu2 = u.reshape(-1, na, 2).repeat(1, na, 1)
    tv1 = torch.kron(v.reshape(-1, na, 2), torch.ones(1, na, 1, device=device))
    tv2 = v.reshape(-1, na, 2).repeat(1, na, 1)
    du  = (tu1 - tu2).norm(p=2, dim=2)
    dv  = (tv1 - tv2).norm(p=2, dim=2)
    du  = torch.mean(torch.kthvalue(du, na + 1, 1, True)[0])
    dv  = torch.mean(torch.kthvalue(dv, na + 1, 1, True)[0])
    return torch.abs(du-dv)

def LH_loss(dH):
    return dH.pow(2).mean()

def LV_loss(u):
    return 1 / ((u[:, 2*int(u.shape[1]/4):]).pow(2).mean() + 1e-5)

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

        self.Aq               = nn.Parameter(1.0*torch.randn(r, d))
        self.Ak               = nn.Parameter(1.0*torch.randn(r, d))
        self.Av               = nn.Parameter(1.0*torch.randn(r, d))
        self.Aq1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Aq5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av5              = nn.Parameter(1.0*torch.randn(r, r))

        self.Ao  = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao1 = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao5 = nn.Parameter(1.0*torch.randn(h, r))

    def forward(self, x, L):

        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        del R22, R21, R12, R11
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        R   = R ** 2

        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = int(torch.sqrt(torch.as_tensor(self.h)))
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        del Rupper, Rdiag
        torch.cuda.synchronize()

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
        self.activation_softA = nn.Softmax(dim=1)
        self.activation_swish = nn.SiLU()
        self.activation_sigmo = nn.Sigmoid()
        self.r                = r
        self.d                = d
        self.h                = h

        self.Aq               = nn.Parameter(1.0*torch.randn(r, d))
        self.Ak               = nn.Parameter(1.0*torch.randn(r, d))
        self.Av               = nn.Parameter(1.0*torch.randn(r, d))
        self.Aq1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Aq5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av5              = nn.Parameter(1.0*torch.randn(r, r))

        self.Ao  = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao1 = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao5 = nn.Parameter(1.0*torch.randn(h, r))

    def forward(self, x, L):

        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

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

        del Q, K, V, o
        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        # j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        # j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j12 = (J**2).sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -(J**2).sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
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

        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        # j12 = J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        # j21 = -J.sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j12 = (J**2).sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -(J**2).sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        J   = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J22, J21, J12, J11
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

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

        self.Aq               = nn.Parameter(1.0*torch.randn(r, d))
        self.Ak               = nn.Parameter(1.0*torch.randn(r, d))
        self.Av               = nn.Parameter(1.0*torch.randn(r, d))
        self.Aq1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av1              = nn.Parameter(1.0*torch.randn(r, r))
        self.Aq5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Ak5              = nn.Parameter(1.0*torch.randn(r, r))
        self.Av5              = nn.Parameter(1.0*torch.randn(r, r))

        self.Ao  = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao1 = nn.Parameter(1.0*torch.randn(r, r))
        self.Ao5 = nn.Parameter(1.0*torch.randn(h, r))

    def forward(self, x, L):
        l = 2
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

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

        del Q, K, V, o
        M11 = torch.kron((M[:, 0:5,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 5:10,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 10:15,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 20:25, :] ** 2).sum(1)

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

        del Q, K, V, o
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        M11 = torch.kron((M[:, 0:5,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 5:10,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 10:15,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 20:25, :] ** 2).sum(1)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        del Mupper11, Mupper12, Mupper21, Mupper22
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        q = x[:, :4, :].transpose(1, 2).reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2) + Mpp.sum(1).unsqueeze(1)

class learnFlocking(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-12
        self.R       = Att_R(8, 4, 4 * 4, device=self.device).to(self.device)
        self.J       = Att_J(8, 4, 4 * 4, device=self.device).to(self.device)
        self.H       = Att_H(8, 6, 5 * 5, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def dHdx(self, inputs):
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        L[:, :, 4::6]   = Q
        L               = L.reshape(-1, 6 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)

        stateH          = torch.zeros((inputs.shape[0] * self.na, 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        stateH = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)
        return dHdx

    def flocking_dynamics(self, t, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        # L               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        # L[:, :, 4::6]   = Q
        # L               = L.reshape(-1, 6 * self.na)
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)
        # state = torch.zeros((vel.shape[0], 6 * self.na), device=self.device)

        # state[:, 0::6] = pos[:, 0::2]
        # state[:, 1::6] = pos[:, 1::2]
        # state[:, 2::6] = vel[:, 0::2]
        # state[:, 3::6] = vel[:, 1::2]
        # state[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        # state[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]
        # state[:, 4::4] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        # state[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)

        # state                                        = state.reshape(-1, self.na, self.na, 6)
        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4]  = stateD
        # state[:, range(self.na), range(self.na), 4:5] = torch.zeros(state.shape[0], self.na, 1, device=self.device)
        # state                                         = state.reshape(-1, 6 * self.na)
        state                                         = state.reshape(-1, 4 * self.na)

        state = L * state

        # R = self.R.forward(state.reshape(-1, self.na, 6).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))

        # J = self.J.forward(state.reshape(-1, self.na, 6).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))

        R = self.R.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        #
        J = self.J.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        stateH = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6]   = Q
        LLL               = LLL.reshape(-1, 6 * self.na)
        stateH            = LLL * stateH
        # stateH            = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        # H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))
        H = self.H.forward(inputsL.to(torch.float32), LLL.reshape(-1, self.na, 6).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        del stateD, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputsL, J, R
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

    def flocking_dynamics_assess(self, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        # L               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        # L[:, :, 4::6]   = Q
        # L               = L.reshape(-1, 6 * self.na)
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)
        # state = torch.zeros((vel.shape[0], 6 * self.na), device=self.device)

        # state[:, 0::6] = pos[:, 0::2]
        # state[:, 1::6] = pos[:, 1::2]
        # state[:, 2::6] = vel[:, 0::2]
        # state[:, 3::6] = vel[:, 1::2]
        # state[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        # state[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]
        # state[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        # state[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)

        # state                                        = state.reshape(-1, self.na, self.na, 6)
        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4]  = stateD
        # state[:, range(self.na), range(self.na), 4:5] = torch.zeros(state.shape[0], self.na, 1, device=self.device)
        # state                                         = state.reshape(-1, 6 * self.na)
        state                                         = state.reshape(-1, 4 * self.na)

        state = L * state

        # R = self.R.forward(state.reshape(-1, self.na, 6).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))

        # J = self.J.forward(state.reshape(-1, self.na, 6).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))

        R = self.R._assess(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        #
        J = self.J._assess(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        stateH = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL               = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6]   = Q
        LLL               = LLL.reshape(-1, 6 * self.na)
        stateH            = LLL * stateH
        # stateH            = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        # H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 6).transpose(1, 2))
        H = self.H._assess(inputsL.to(torch.float32), LLL.reshape(-1, self.na, 6).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        del stateD, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputsL, J, R
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics_assess(self, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics_assess(self, inputs):
        dd = self.leader_dynamics_assess(inputs)
        da = self.flocking_dynamics_assess(inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def assess(self, inputs, simulation_time, step_size):
        outputs_2 = torch.zeros(len(simulation_time), inputs.shape[1], device=self.device)
        for i in range(len(simulation_time)):
            outputs_2[i, :] = inputs + step_size * self.overall_dynamics_assess(inputs)
            inputs = torch.clone(outputs_2[i, :].unsqueeze(0))
            # Free unused memory
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                libc.malloc_trim()
        return outputs_2

class Att_RU(nn.Module):
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

        del Q, K, V, o
        # R11 = R[:, 0,   :]
        # R12 = R[:, 1,   :]
        # R21 = R[:, 2,  :]
        R11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        R12 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        R21 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        R22 = R[:, 3, :]
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

        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = 4
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        return R

    def _assess(self, x, L):
        l  = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        R  = torch.zeros(x.shape[0], l * l, na,  device=self.device)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        R   = torch.cat((torch.cat((R11, R21), dim=1), torch.cat((R12, R22), dim=1)), dim=2)
        del R22, R21, R12, R11
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        R   = R ** 2

        Rupper = R + R.transpose(1, 2)
        Rdiag  = torch.clone(Rupper)

        l = int(torch.sqrt(torch.as_tensor(self.h)))
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        del Rupper, Rdiag
        torch.cuda.synchronize()

        return R

class Att_JU(nn.Module):
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

        del Q, K, V, o
        J11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        J22 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        j12 = (J**2).sum(1).sum(1).reshape(int(x.shape[0] / x.shape[2]), na)
        j21 = -torch.clone(j12)
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
        # J = J ** 2
        # J = J - J.transpose(1, 2)

        return J

    def _assess(self, x, L):
        l  = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        J  = torch.zeros(x.shape[0], l * l, na,  device=self.device)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        J   = torch.cat((torch.cat((J11, J21), dim=1), torch.cat((J12, J22), dim=1)), dim=2)
        del J22, J21, J12, J11
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        J   = J ** 2
        J   = J - J.transpose(1, 2)

        return J

class Att_HU(nn.Module):
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
        q = torch.cat((x[:, 0:2, :], x[:, 4:6, :]), dim=1)
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        Q = self.activation_sigmo(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), q)).transpose(1, 2)
        K = self.activation_sigmo(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), q))
        V = self.activation_sigmo(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), q)).transpose(1, 2)

        # Q = self.activation_sigmo(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)
        # K = self.activation_sigmo(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        # V = self.activation_sigmo(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x)).transpose(1, 2)

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
        #
        o = torch.bmm(self.activation_soft(torch.bmm(Q, K) / torch.sqrt(numNeighbors)).to(torch.float32), V).transpose(1, 2)
        M = self.activation_swish(torch.bmm(self.Ao5.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o))

        del Q, K, V, o
        M11 = (M[:, 0:4, :]**2).sum(1)
        # M12 = M[:, 1, :]
        # M21 = M[:, 2, :]
        # M11 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        M12 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        M21 = torch.zeros((int(x.shape[0] / x.shape[2]), na, na), device=self.device)
        M22 = M[:, 4, :]** 2
        # Mpp = M[:, 4, :]
        M11 = M11.reshape(int(x.shape[0] / x.shape[2]), na, na)
        M12 = M12.reshape(int(x.shape[0] / x.shape[2]), na, na)
        M21 = M21.reshape(int(x.shape[0] / x.shape[2]), na, na)
        M22 = M22.reshape(int(x.shape[0] / x.shape[2]), na, na)
        M11 = torch.kron(M11, torch.eye(2, device=self.device).unsqueeze(0))
        M12 = torch.kron(M12, torch.eye(2, device=self.device).unsqueeze(0))
        M21 = torch.kron(M21, torch.eye(2, device=self.device).unsqueeze(0))
        M22 = torch.kron(M22, torch.eye(2, device=self.device).unsqueeze(0))
        R = torch.cat((torch.cat((M11, M21), dim=1), torch.cat((M12, M22), dim=1)), dim=2)
        # R = R ** 2

        Rupper = R + R.transpose(1, 2)
        Rdiag = torch.clone(Rupper)

        l = 4
        Rdiag[:, range(l * na), range(l * na)] = torch.zeros(l * na, device=self.device)

        Rout = torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * Rupper

        R = Rout + torch.eye(l * na, device=self.device).unsqueeze(dim=0).repeat(int(x.shape[0] / x.shape[2]), 1, 1) * torch.sum(Rdiag, 2).unsqueeze(2) - Rdiag

        p = x.reshape(-1, na, x.shape[1], na).transpose(2, 3)[:, range(na), range(na), 2:4].reshape(-1, 1, 2 * na)

        return torch.bmm(p, torch.bmm(R[:, 2 * na:, 2 * na:], p.transpose(1, 2))).sum(2) \
               + torch.bmm(torch.ones(R.shape[0], 1, 2 * na, device=self.device), torch.bmm(R[:, :2 * na, :2 * na], torch.ones(R.shape[0], 2 * na, 1, device=self.device))).sum(2)

    def _assess(self, x, L):
        l = int(torch.sqrt(torch.as_tensor(self.h)))
        na = x.shape[2]
        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1) + 1
        M  = torch.zeros(x.shape[0], l * l, na,  device=self.device)
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        M11 = torch.kron((M[:, 0:4,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 4:8,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 8:12,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 12:16, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)

        q = x.transpose(1, 2).reshape(-1, 1, 4 * na)

        return torch.bmm(q, torch.bmm(M, q.transpose(1, 2))).sum(2)

class learnFlockingU(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = Att_RU(8, 4, 4, device=self.device).to(self.device)
        self.J       = Att_JU(8, 4, 1, device=self.device).to(self.device)
        # self.H       = Att_HU(8, 6, 4 * 4, device=self.device).to(self.device)
        self.H       = Att_HU(8, 4, 5, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def flocking_dynamics(self, t, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, ::4]  = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = stateD
        state                                        = state.reshape(-1, 4 * self.na)

        state = L * state

        R = self.R.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        J = self.J.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))

        stateH = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        # LLL[:, :, 4::6] = Q
        LLL = LLL.reshape(-1, 6 * self.na)
        stateH = LLL * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 6).transpose(1, 2).data, requires_grad=True)
        H       = self.H.forward(inputsL.to(torch.float32), LLL.reshape(-1, self.na, 6).transpose(1, 2))
        del stateD, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        Hgrad   = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH      = Hgrad[0]
        dHq     = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq     = dHq[:, range(self.na), range(self.na), :]
        dHp     = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp     = dHp[:, range(self.na), range(self.na), :]
        dHdx    = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputsL, J, R
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:]

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd     = self.leader_dynamics(t, inputs)
        da     = self.flocking_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

    def flocking_dynamics_assess(self, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 5), device=self.device))
        L[:, :, 4::5]   = Q
        L               = L.reshape(-1, 5 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2),
                          torch.ones((1, self.na, 1), device=self.device))
        pos = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2),
                          torch.ones((1, self.na, 1), device=self.device))
        vel = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 5 * self.na), device=self.device)

        state[:, ::5] = pos[:, 0::2]
        state[:, 1::5] = pos[:, 1::2]
        state[:, 2::5] = vel[:, 0::2]
        state[:, 3::5] = vel[:, 1::2]
        state[:, 4::5] = (pos.reshape(-1, self.na, 2).norm(p=2, dim=2) + self.epsilon).pow(-1 / 4).unsqueeze(2).reshape(
            -1, self.na)

        state = state.reshape(-1, self.na, self.na, 5)
        state[:, range(self.na), range(self.na), :4] = stateD
        state[:, range(self.na), range(self.na), 4:] = torch.zeros(state.shape[0], self.na, 1, device=self.device)
        state = state.reshape(-1, 5 * self.na)

        state = L * state

        R = self.R._assess(state.reshape(-1, self.na, 5).transpose(1, 2).to(torch.float32),
                           L.reshape(-1, self.na, 5).transpose(1, 2))

        J = self.J._assess(state.reshape(-1, self.na, 5).transpose(1, 2).to(torch.float32),
                           L.reshape(-1, self.na, 5).transpose(1, 2))

        L = torch.kron(LL, torch.ones(1, 1, 4, device=self.device)).reshape(-1, 4 * self.na)
        state = torch.zeros((inputs.shape[0], 4 * self.na), device=self.device)
        state[:, 0::4] = stateD[:, :, 0]
        state[:, 1::4] = stateD[:, :, 1]
        state[:, 2::4] = stateD[:, :, 2]
        state[:, 3::4] = stateD[:, :, 3]
        state = L * torch.kron(state, torch.ones(self.na, 1, device=self.device))

        inputsL = Variable(state.reshape(-1, self.na, 4).transpose(1, 2).data, requires_grad=True)
        H = self.H._assess(inputsL.to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        del stateD, state, L, pos, pos1, pos2, vel, vel1, vel2, LL
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.unsqueeze(2)).squeeze(2)

        del dHdx, dHp, dHq, dH, Hgrad, H, inputsL, J, R
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:]

    def leader_dynamics_assess(self, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics_assess(self, inputs):
        dd = self.leader_dynamics_assess(inputs)
        da = self.flocking_dynamics_assess(inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def assess(self, inputs, simulation_time, step_size):
        outputs_2 = torch.zeros(len(simulation_time), inputs.shape[1], device=self.device)
        for i in range(len(simulation_time)):
            outputs_2[i, :] = inputs + step_size*self.overall_dynamics_assess(inputs)
            inputs          = torch.clone(outputs_2[i, :].unsqueeze(0))
            # Free unused memory
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

        self.A                = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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
        self.A                = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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

        self.A = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, 0:5,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 5:10,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 10:15,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 20:25, :] ** 2).sum(1)

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

class learnFlockingMLP(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-12
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

    def dHdx(self, inputs):
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        stateH          = torch.zeros((inputs.shape[0] * self.na, 4 * self.na), device=self.device)
        stateH[:, 0::4] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::4] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::4] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::4] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 4).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)
        return dHdx

    def flocking_dynamics(self, t, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = stateD
        state                                        = state.reshape(-1, 4 * self.na)

        state = L * state

        R = self.R.forward(state)

        J = self.J.forward(state)

        stateH = torch.zeros((state.shape[0], 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        LLL = torch.kron(LL, torch.ones((1, 1, 6), device=self.device))
        LLL[:, :, 4::6] = Q
        LLL = LLL.reshape(-1, 6 * self.na)
        stateH = LLL * stateH

        del LL, inputsD, inputsL, inputs, pos, pos1, pos2, vel, vel1, vel2, stateD, Q, LLL

        inputsL = Variable(stateH.data, requires_grad=True)
        H       = self.H.forward(inputsL.to(torch.float32))
        Hgrad   = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH      = Hgrad[0].reshape(state.shape[0], self.na, 6)
        dHq     = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ     = dHq[:, range(self.na), range(self.na), :]
        dHp     = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP     = dHp[:, range(self.na), range(self.na), :]
        dHdx    = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
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

        self.A               = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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
        self.A                = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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

        self.A = nn.Parameter(0.01*torch.randn(d, h))

    def forward(self, x):
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, 0:5,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 5:10,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 10:15,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 20:25, :] ** 2).sum(1)

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

class learnFlockingGNN(nn.Module):

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

    def dHdx(self, inputs):
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        stateH          = torch.zeros((inputs.shape[0] * self.na, 4 * self.na), device=self.device)
        stateH[:, 0::4] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::4] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::4] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::4] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 4).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)
        return dHdx

    def flocking_dynamics(self, t, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state                                        = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = stateD
        state                                        = state.transpose(2, 3).transpose(1, 2)

        stateH = torch.zeros((inputs.shape[0], 4, self.na, self.na), device=self.device)
        stateH[:, 0, :, :] = torch.bmm(LL, state[:, 0, :, :])
        stateH[:, 1, :, :] = torch.bmm(LL, state[:, 1, :, :])
        stateH[:, 2, :, :] = torch.bmm(LL, state[:, 2, :, :])
        stateH[:, 3, :, :] = torch.bmm(LL, state[:, 3, :, :])
        stateH             = stateH.transpose(1, 2).transpose(2, 3).reshape(-1, 4 * self.na)

        R = self.R.forward(stateH)

        J = self.J.forward(stateH)

        stateH = torch.zeros((state.shape[0] * self.na, 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        stateH          = stateH.reshape(-1, self.na, self.na, 6)
        stateH          = stateH.transpose(2, 3).transpose(1, 2)
        stateH[:, 0, :, :] = torch.bmm(LL, stateH[:, 0, :, :])
        stateH[:, 1, :, :] = torch.bmm(LL, stateH[:, 1, :, :])
        stateH[:, 2, :, :] = torch.bmm(LL, stateH[:, 2, :, :])
        stateH[:, 3, :, :] = torch.bmm(LL, stateH[:, 3, :, :])
        stateH[:, 4, :, :] = torch.bmm(Q, stateH[:, 4, :, :])
        stateH[:, 5, :, :] = torch.bmm(LL, stateH[:, 5, :, :])
        stateH             = stateH.transpose(1, 2).transpose(2, 3).reshape(-1, 6 * self.na)

        del state, LL, inputsD, inputsL, inputs, pos, pos1, pos2, vel, vel1, vel2, stateD, Q

        inputsL = Variable(stateH.data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0].reshape(stateH.shape[0], self.na, 6)
        dHq = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
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
        self.Aq               = nn.Parameter(0.6*torch.randn(r, d))
        self.Ak               = nn.Parameter(0.6*torch.randn(r, d))
        self.Av               = nn.Parameter(0.6*torch.randn(r, d))
        self.Ao               = nn.Parameter(0.6*torch.randn(1, r))

    def forward(self, x, L):

        numNeighbors = L[:, 0, :].ge(1).double().sum(dim=1).view(-1, 1, 1) + 1

        Q = self.activation_tanh(torch.bmm(self.Aq.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        K = self.activation_tanh(torch.bmm(self.Ak.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))
        V = self.activation_tanh(torch.bmm(self.Av.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x))

        o = torch.bmm(self.activation_soft(torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(numNeighbors)).to(torch.float32), V)
        R = self.activation_tanh(torch.bmm(self.Ao.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), o)).squeeze(dim=1)
        del Q, K, V, o
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

        self.A               = nn.Parameter(0.6*torch.randn(d, h))

    def forward(self, x):
        R  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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
        self.A                = nn.Parameter(0.6*torch.randn(d, h))

    def forward(self, x):
        J  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
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

        self.A = nn.Parameter(0.6*torch.randn(d, h))

    def forward(self, x):
        M  = self.activation_swish(torch.bmm(x.unsqueeze(dim=1), self.A.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)))
        l  = 5
        na = int(x.shape[1]/torch.as_tensor(6))
        M  = M.reshape(x.shape[0], l * l, na)

        M11 = torch.kron((M[:, 0:5,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M12 = torch.kron((M[:, 5:10,   :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M21 = torch.kron((M[:, 10:15,  :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        M22 = torch.kron((M[:, 15:20, :] ** 2).sum(1), torch.ones(1, 2, device=self.device))
        Mpp = (M[:, 20:25, :] ** 2).sum(1)

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

class learnFlockingGNNSA(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.device  = parameters["device"]
        self.na      = parameters['na']
        self.r_max   = parameters['r']/2
        self.epsilon = 1e-5
        self.R       = GNN_R(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.J       = GNN_J(4 * self.na, 4 * 4 * self.na, device=self.device).to(self.device)
        self.H       = GNN_H(6 * self.na, 5 * 5 * self.na, device=self.device).to(self.device)
        self.mask    = Att_mask(8, 4, device=self.device).to(self.device)

    def laplacian(self, q_agents):
        Q1 = q_agents.reshape(q_agents.shape[0], -1, 2).repeat(1, self.na, 1)
        Q2 = torch.kron(q_agents.reshape(q_agents.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        Q  = (Q1 - Q2).norm(p=2, dim=2).reshape(q_agents.shape[0], self.na, self.na)
        L  = Q.le(self.r_max).float()
        L1 = L * torch.sigmoid(-(2.0)*(Q - self.r_max))
        L2 = L * torch.sigmoid((2.0)*(Q - self.r_max))
        return L1, L2

    def dHdx(self, inputs):
        LL, Q           = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        stateH          = torch.zeros((inputs.shape[0] * self.na, 4 * self.na), device=self.device)
        stateH[:, 0::4] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::4] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::4] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::4] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH = L * stateH

        inputsL = Variable(stateH.reshape(-1, self.na, 4).transpose(1, 2).data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0]
        dHq = dH[:, :2, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHq = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, 2:4, :].reshape(-1, self.na, 2, self.na).transpose(2, 3)
        dHp = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHq.reshape(-1, 2 * self.na), dHp.reshape(-1, 2 * self.na)), dim=1)
        return dHdx

    def flocking_dynamics(self, t, inputs):
        LL, Q          = self.laplacian(inputs[:, :2 * self.na])
        inputs          = nn.functional.normalize(inputs, p=2, dim=1)
        L               = torch.kron(LL, torch.ones((1, 1, 4), device=self.device))
        L               = L.reshape(-1, 4 * self.na)
        inputsL         = inputs[:, :4 * self.na]
        inputsD         = (inputs[:, :4 * self.na] - inputs[:, 4 * self.na:])
        stateD          = torch.zeros((inputs.shape[0], self.na, 4), device=self.device)
        stateD[:, :, 0] = inputsD[:, 0:2 * self.na:2]
        stateD[:, :, 1] = inputsD[:, 1:2 * self.na:2]
        stateD[:, :, 2] = inputsD[:, 2 * self.na + 0::2]
        stateD[:, :, 3] = inputsD[:, 2 * self.na + 1::2]

        pos1 = inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        pos2 = torch.kron(inputsL[:, :2 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        pos  = (pos1 - pos2).reshape(inputsL.shape[0] * self.na, -1)
        vel1 = inputsL[:, 2 * self.na:6 * self.na].reshape(inputsL.shape[0], -1, 2).repeat(1, self.na, 1)
        vel2 = torch.kron(inputsL[:, 2 * self.na:4 * self.na].reshape(inputsL.shape[0], -1, 2), torch.ones((1, self.na, 1), device=self.device))
        vel  = (vel1 - vel2).reshape(inputsL.shape[0] * self.na, -1)

        state = torch.zeros((vel.shape[0], 4 * self.na), device=self.device)

        state[:, 0::4] = pos[:, 0::2]
        state[:, 1::4] = pos[:, 1::2]
        state[:, 2::4] = vel[:, 0::2]
        state[:, 3::4] = vel[:, 1::2]

        state = state.reshape(-1, self.na, self.na, 4)
        state[:, range(self.na), range(self.na), :4] = stateD
        state = state.transpose(2, 3).transpose(1, 2)

        mask = self.mask.forward(state.reshape(-1, self.na, 4).transpose(1, 2).to(torch.float32), L.reshape(-1, self.na, 4).transpose(1, 2))
        maksL = mask * LL
        maksQ = mask * Q

        stateH = torch.zeros((inputs.shape[0], 4, self.na, self.na), device=self.device)
        stateH[:, 0, :, :] = torch.bmm(maksL, state[:, 0, :, :])
        stateH[:, 1, :, :] = torch.bmm(maksL, state[:, 1, :, :])
        stateH[:, 2, :, :] = torch.bmm(maksL, state[:, 2, :, :])
        stateH[:, 3, :, :] = torch.bmm(maksL, state[:, 3, :, :])
        stateH = stateH.transpose(1, 2).transpose(2, 3).reshape(-1, 4 * self.na)

        R = self.R.forward(stateH)

        J = self.J.forward(stateH)

        stateH = torch.zeros((stateH.shape[0], 6 * self.na), device=self.device)
        stateH[:, 0::6] = torch.kron(stateD[:, :, 0], torch.ones(self.na, 1, device=self.device))
        stateH[:, 1::6] = torch.kron(stateD[:, :, 1], torch.ones(self.na, 1, device=self.device))
        stateH[:, 2::6] = torch.kron(stateD[:, :, 2], torch.ones(self.na, 1, device=self.device))
        stateH[:, 3::6] = torch.kron(stateD[:, :, 3], torch.ones(self.na, 1, device=self.device))
        stateH[:, 4::6] = ((pos.reshape(-1, self.na, 2)).norm(p=2, dim=2) + self.epsilon).pow(-1).unsqueeze(2).reshape(-1, self.na)
        stateH[:, 5::6] = (pos.reshape(-1, self.na, 2)).norm(p=2, dim=2).unsqueeze(2).reshape(-1, self.na)
        stateH = stateH.reshape(-1, self.na, self.na, 6)
        stateH = stateH.transpose(2, 3).transpose(1, 2)
        stateH[:, 0, :, :] = torch.bmm(maksL, stateH[:, 0, :, :])
        stateH[:, 1, :, :] = torch.bmm(maksL, stateH[:, 1, :, :])
        stateH[:, 2, :, :] = torch.bmm(maksL, stateH[:, 2, :, :])
        stateH[:, 3, :, :] = torch.bmm(maksL, stateH[:, 3, :, :])
        stateH[:, 4, :, :] = torch.bmm(maksQ, stateH[:, 4, :, :])
        stateH[:, 5, :, :] = torch.bmm(maksL, stateH[:, 5, :, :])
        stateH = stateH.transpose(1, 2).transpose(2, 3).reshape(-1, 6 * self.na)

        del state, LL, inputsD, inputsL, inputs, pos, pos1, pos2, vel, vel1, vel2, stateD, Q

        inputsL = Variable(stateH.data, requires_grad=True)
        H = self.H.forward(inputsL.to(torch.float32))
        Hgrad = torch.autograd.grad(H.sum(), inputsL, only_inputs=True, create_graph=True)
        dH = Hgrad[0].reshape(stateH.shape[0], self.na, 6)
        dHq = dH[:, :, 0:2].reshape(-1, self.na, self.na, 2)
        dHQ = dHq[:, range(self.na), range(self.na), :]
        dHp = dH[:, :, 2:4].reshape(-1, self.na, self.na, 2)
        dHP = dHp[:, range(self.na), range(self.na), :]
        dHdx = torch.cat((dHQ.reshape(-1, 2 * self.na), dHP.reshape(-1, 2 * self.na)), dim=1)

        dx = torch.bmm(J.to(torch.float32) - R.to(torch.float32), dHdx.to(torch.float32).unsqueeze(2)).squeeze(2)

        del R, J, dHdx, dH, dHq, dHQ, dHP

        # Free unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dx[:, :2 * self.na], dx[:, 2 * self.na:],

    def leader_dynamics(self, t, inputs):
        return inputs[:, 6 * self.na:], torch.zeros((inputs.shape[0], 2 * self.na), device=self.device)

    def overall_dynamics(self, t, inputs):
        dd = self.leader_dynamics(t, inputs)
        da = self.flocking_dynamics(t, inputs)
        return torch.cat((da[0] + dd[0], da[1] + dd[1], dd[0], dd[1]), dim=1)

    def forward(self, inputs, simulation_time, step_size):
        outputs_2 = odeint(self.overall_dynamics, inputs, simulation_time.to(self.device), method='euler', options={'step_size': step_size})
        return outputs_2

def generate_agents(na):
    return 4.0 * torch.rand(2 * na), torch.zeros(2 * na)

def generate_agents_test(na):
    return 0.5 * torch.randn(2 * na), torch.zeros(2 * na)

def generate_leader(na):
    return torch.zeros(2 * na), torch.zeros(2 * na)

def get_cmap(n):
    return plt.cm.get_cmap('hsv', n+1)
