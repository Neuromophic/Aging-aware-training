import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import config

class PNNLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, aging_generator, xpu='cpu'):
        '''
        Generate a PNN layer
        :param n_in: number of input of the layer
        :param n_out: number of output of the layer
        '''
        super(PNNLayer, self).__init__()
        # initialize theta
        # theta is [n_out , n_in + 2] vector. Plus two -> 1 for bias, 1 for decouple
        # a row of theta consists of [theta_1, theta_2, ..., theta_{n_in}, theta_b, theta_d]
        theta = torch.rand([n_out, n_in + 2])
        theta[:, -1] = theta[:, -1] * 100
        theta[:, -2] = 0.1788 / (1 - 0.1788) * (torch.sum(torch.abs(theta[:, :-3]), axis=1) + torch.abs(theta[:, -1]))
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

        # initializ CPU/GPU
        self.xpu = xpu

        # initialize time
        self.t = [0]

        # initialize aging model
        self.totalnum = n_out * (n_in + 2)
        self.aging_generator = aging_generator

        # for straight throught estimator
        self.st = g_straight_through.apply

    def GenerateAgingModel(self, M=1):
        self.models = [self.aging_generator.get_models(self.totalnum) for o in range(M)]

    @property
    def theta(self):
        '''
        put theta into straight through function
        '''
        return self.st(self.theta_)

    def theta_aged(self, model):
        '''
        multiply theta_initial with the aging decay coefficient
        '''
        s = list(self.theta.shape)
        s.insert(0, len(self.t))  # s = [K, n_out, n_in]

        # generate aging decay coefficient [K, n_out, n_in]
        aging_decay = torch.tensor(np.array([m(self.t) for m in model])).T.reshape(s)

        theta_temp = self.theta * aging_decay.to(self.xpu)  # shape [K, n_out, n_in]
        return theta_temp

    @property
    def g(self):
        '''
        Get the absolute value of the surrogate conductance aged theta
        :return: absolute(theta)
        '''
        g = torch.cat([self.theta_aged(model)[None, :, :, :] for model in self.models])
        return g.abs()

    def inv(self, x):
        '''
        Quasi-negative value of x
        In w*x, when w < 0, it is not implementable as the resistor has no negative resistance.
        We conver the problem to (-w) * (-x) =: g * inv(x)
        :param x: values to be calculated
        :return: inv(x)
        '''
        # different constants for each columns later (variation)
        return 0.104 - 0.899 * torch.tanh((x + 0.056) * 3.858)

    def activate(self, z):
        '''
        activation function of PNN
        :param z: parameter after MAC
        :return: values after activation
        '''
        return 0.134 + 0.962 * torch.tanh((z - 0.183) * 24.10)

    def mac(self, a):
        '''
        convert g to w, calculate multiply-accumulate considering inv(a)
        :param a: input of the layer
        :return: output after MAC
        '''
        self.M = a.shape[0]  # number of models
        self.K = a.shape[1]  # number of time points
        E = a.shape[2]  # number of examples
        m, n = self.theta.shape  # number of output and input neurons

        # enlarge for parameter b and d
        # a changes from [M, K, E, n_in] to [M, K, E, n_in+2]
        a = torch.cat(
            [a, torch.ones(self.M, self.K, E, 1).to(self.xpu), torch.zeros(self.M, self.K, E, 1).to(self.xpu)], dim=3)

        # calculate the negative a, i.e. inv(a)
        InvX = self.inv(a)
        # but coefficient for gd is always 0
        InvX[:, :, :, -1] = 0

        # convert g to weights
        W = self.g / torch.sum(self.g, axis=3, keepdim=True)  # [M, K, n_out, n_in+2]

        # repeat input matrix m times to m layers, each layer is for one combination of theta
        # [M, K, n_out, E, n_in+2]
        X_repeat = a.repeat(m, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
        InvX_repeat = InvX.repeat(m, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)

        # find which element assigned to which operator, i.e., a = a & a = inv(a)
        # size: [M, K, n_out, 1, n_in+2]
        postheta = (self.theta >= 0).int().repeat(self.M, self.K, 1, 1)[None, :, :, :, :].permute(1, 2, 3, 0, 4)
        negtheta = (self.theta < 0).int().repeat(self.M, self.K, 1, 1)[None, :, :, :, :].permute(1, 2, 3, 0, 4)

        # combine matrix with x and with inv(x) together
        # [M, K, n_out, E, n_in+2] * [M, K, n_out, 1, n_in+2] = [M, K, n_out, E, n_in+2]
        pos_X_repeat = torch.mul(X_repeat, postheta)
        neg_X_repeat = torch.mul(InvX_repeat, negtheta)
        X_sum = pos_X_repeat + neg_X_repeat  # [M, K, n_out, E, n_in+2]

        # each row of weights was put to each corresponding layer
        W_temp = W[:, :, :, :, None].type(torch.float32)  # [M, K, n_out, n_in+2, 1]

        # multiply x or inv(x) with w, each layer is corresponding to one output neuron
        z = torch.matmul(X_sum, W_temp)[:, :, :, :, 0]  # [M, K, n_out, E]
        # transpose
        z_s = z.permute(0, 1, 3, 2)  # [M, K, E, n_out]
        return z_s

    def forward(self, a_previous):
        '''
        forward propagation: MAC and activation
        :param a: input of the layer
        :return: output of the layer
        '''
        z_new = self.mac(a_previous)
        a_new = self.activate(z_new)
        return a_new


class g_straight_through(torch.autograd.Function):
    '''
    straight through is a special function, whoes forward and backward propagation are different.
    '''

    @staticmethod
    def forward(ctx, theta, g_min=0.01):
        '''
        forward propagation is a piecewise linear function
        '''
        theta_temp = theta.clone()
        theta_temp[theta_temp.abs() < g_min] = 0.
        ctx.save_for_backward(theta_temp)
        return theta_temp

    @staticmethod
    def backward(ctx, grad_output):
        '''
        backward is a continuous linear function
        '''
        return grad_output


def LossFunction(prediction, label, dimension=None, m=config.m, T=config.T):
    '''
    loss function for vectorized pNN
    :param prediction: predictions from pNN
    :param label: label without vectorization
    :param m: sensing margin
    :param T: sensing-related hyperparameter
    :return: loss
    '''

    # dimensionality of predictions, i.e., num of models and timings
    M = prediction.shape[0]
    K = prediction.shape[1]

    # vectorize labels
    label = label.reshape(-1, 1).repeat(M, K, 1, 1)  # [M, K, E, N_Class]

    # find output of right classes
    fy = prediction.gather(3, label)[:, :, :, 0]  # [M, K, E]

    # find largest output in wrong classes by
    # 1. setting output of right class to -Inf
    fny = prediction.clone()
    fny = fny.scatter_(3, label, -10 ** 10)  # [M, K, E, N_Class]
    # 2. finding the largest output
    fnym = torch.max(fny, axis=3).values  # [M, K, E]

    # losses of E examples in M*E aging-situations
    l = torch.max(m + T - fy, torch.tensor(0)) + torch.max(m + fnym, torch.tensor(0))  # [M, K, E]
    
    if dimension is None:
        # average all losses as the final loss
        L = torch.mean(l)  # L is a single value
    else:
        # this loss if for analysing loss and time, not for backpropagation
        L = torch.mean(l, dim=dimension)
    return L


def MakeModel(m, M=1):
    '''
    let pNN generate stochastic aging models
    :param m: pNN
    :param M: number of aging models
    '''
    if isinstance(m, PNNLayer):
        m.GenerateAgingModel(M)


def SetTime(m, t):
    '''
    let pNN calculate the aging decay for generated stochastic aging models at t timings
    :param m: pNN
    :param t: list of timings
    '''
    if isinstance(m, PNNLayer):
        m.t = t


def SetDevice(m, xpu):
    '''
    move data to devices
    :param m: pNN
    :param xpu: cpu/gpu
    '''
    if isinstance(m, PNNLayer):
        m.xpu = xpu
