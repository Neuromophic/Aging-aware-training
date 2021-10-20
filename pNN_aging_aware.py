import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt



class PNNLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, aging_generator):
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
        theta[:,-1] = theta[:,-1] * 100
        theta[:, -2] = 0.1788 / (1 - 0.1788) * (torch.sum(torch.abs(theta[:, :-3]), axis=1) + torch.abs(theta[:,-1]))
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)
        
        # initialize aging model
        self.t = 0
        self.model = aging_generator.get_models(n_out*(n_in+2))
        
        # for straight throught estimator
        self.st = g_straight_through.apply
        
    @property
    def theta(self):
        '''
        put theta into straight through function
        '''
        return self.st(self.theta_aged)
    
    @property
    def theta_aged(self):
        '''
        multiply theta_initial with the aging decay coefficient
        '''
        # generate aging decay coefficient
        aging_decay = torch.tensor([m([self.t]) for m in self.model])
        # multiply them
        s = self.theta_.shape
        theta_temp = self.theta_.clone()
        theta_temp = theta_temp.view(-1,1)
        theta_temp *= aging_decay    
        return theta_temp.view(s)

    @property
    def g(self):
        '''
        Get the absolute value of the surrogate conductance theta
        :return: absolute(theta)
        '''
        g = self.theta
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
        M = a.shape[0]  # number of examples
        m, n = self.g.shape  # number of output and input neurons

        # enlarge for parameter b and d
        a = torch.hstack((a, torch.ones(M, 1), torch.zeros(M, 1)))

        # convert g to weights
        W = self.g / torch.sum(self.g, axis=1).view(-1, 1)

        # calculate the negative a, i.e. inv(a)
        InvX = self.inv(a)

        # repeat input matrix m times to m layers, each layer is for one combination of theta
        X_repeat = a.repeat(m, 1, 1)
        InvX_repeat = InvX.repeat(m, 1, 1)

        # find which element assigned to which operator, i.e., a = a & a = inv(a)
        postheta = (self.theta >= 0) * 1.0
        negtheta = (self.theta < 0) * 1.0

        # each row of theta is assigned to one layer
        pt = postheta.view(m, 1, n)
        nt = negtheta.view(m, 1, n)

        # combine matrix with x and with int(x) together
        pos_X_repeat = torch.mul(X_repeat, pt)
        neg_X_repeat = torch.mul(InvX_repeat, nt)
        X_sum = pos_X_repeat + neg_X_repeat

        # each row of weights was put to each corresponding layer
        W_temp = W.view(m, n, 1)

        # multiply x or inv(x) with w, each layer is corresponding to one output neuron
        z = torch.matmul(X_sum, W_temp)
        # resize
        z_s = z.view(m, M).t()
        return z_s

    def forward(self, a_previous, t):
        '''
        forward propagation: MAC and activation
        :param a: input of the layer
        :return: output of the layer
        '''
        self.t = t
        z_new = self.mac(a_previous)
        a_new = self.activate(z_new)
        return a_new

    def projg(self, g_max):
        '''
        theta > g_max ==> g_max
        -theta < -g_max ==> -g_max
        '''
        theta_temp = self.theta.where(self.theta <= g_max, torch.tensor(g_max))
        theta_temp = theta_temp.where(self.theta >= -g_max, torch.tensor(-g_max))
        self.theta = theta_temp        
    
    
class g_straight_through(torch.autograd.Function):
    '''
    straight through is a special function, whoes forward and backward propagation are different.
    '''
    @staticmethod
    def forward(ctx, theta, g_min=0.01):
        '''
        forward propagation is a piecewise linear function
        '''
        theta = theta.clone()
        theta[theta.abs() < g_min] = 0.
        ctx.save_for_backward(theta)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        '''
        backward is a continuous linear function
        '''
        return grad_output


def LossFunction(prediction, label, m, T):
    label = label.reshape(-1, 1)
    fy = prediction.gather(1, label).reshape(-1, 1)
    fny = prediction.clone()
    fny = fny.scatter_(1, label, -10 ** 10)
    fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
    l = torch.max(m + T - fy, torch.tensor(0)) + torch.max(m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L


