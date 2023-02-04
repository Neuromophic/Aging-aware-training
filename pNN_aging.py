import numpy as np
import torch

class AApLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, aging_generator, args):
        super().__init__()
        self.args = args
        self.device = args.DEVICE
        
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = args.ACT_eta3/(1.-args.ACT_eta3)*(torch.sum(theta[:-2,:], axis=0)+theta[-1,:])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)
        
        # initialize time sampling
        # t = 0 equals nominal training
        self.K = args.K_train
        if args.MODE == 'nominal':
            self.t = torch.tensor([0.])
        else:
            self.t = torch.linspace(0, 1, self.K)
        # initialize aging model
        self.M = args.M_train
        self.aging_generator = aging_generator
        
        # initialization for variation
        self.N = args.N_train
        self.epsilon = args.e_train
    
    @property
    def agingmodels(self):
        return self.aging_generator.get_models(self.M*self.theta_.numel()*self.N) # M aging models for each sampled theta (N variations)

    @property
    def theta_ideal(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()
    
    @property
    def theta(self):
        mean = self.theta_ideal.repeat(self.N, 1, 1)
        variation = (torch.rand(mean.shape)*2. - 1.) * self.epsilon + 1.
        return mean.to(self.device) * variation.to(self.device)
    
    @property
    def theta_aged(self):
        # generate aging decay coefficient [M, K, N, n_in, n_out]
        aging_decay = torch.tensor([m(self.t) for m in self.agingmodels]) # [M*N*n_in*n_out, K]
        aging_decay = aging_decay.reshape(self.M,self.theta.shape[0],self.theta.shape[1],self.theta.shape[2],self.K).permute(0,4,1,2,3)
        # broad casting: [M, K, N, n_in, n_out] * [N, n_in, n_out] -> multiply for the last 2 dimension
        return self.theta * aging_decay.to(self.device)
    
    @property
    def W(self):
        return self.theta_aged.abs() / torch.sum(self.theta_aged.abs(), axis=3, keepdim=True)

    def INV(self, x):
        return -(self.args.NEG_eta1 + self.args.NEG_eta2 * torch.tanh((x - self.args.NEG_eta3) * self.args.NEG_eta4))
    
    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0]  = 0.
        negative = 1. - positive
        # a in [M, K, N, E, n_in]
        a_extend = torch.cat([a,
                              torch.ones( [a.shape[0], a.shape[1], a.shape[2], a.shape[3], 1]).to(self.device),
                              torch.zeros([a.shape[0], a.shape[1], a.shape[2], a.shape[3], 1]).to(self.device)], dim=4)
        a_neg = self.INV(a_extend)
        a_neg[:,:,:,:,-1] = 0.
        z = torch.matmul(a_extend, self.W * positive) + torch.matmul(a_neg, self.W * negative)
        return z
    
    def ACT(self, z):
        return self.args.ACT_eta1 + self.args.ACT_eta2 * torch.tanh((z - self.args.ACT_eta3) * self.args.ACT_eta4)
    
    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        a_new = self.ACT(z_new)
        return a_new
    
    def SetParameter(self, name, value):
        # set time sampling and update K
        if name == 't':
            self.t = value
            self.K = self.t.shape[0]
        # set number of aging-model sampling M
        elif name == 'M':
            self.M = value
        # set number of samples
        elif name == 'N':
            self.N = value
        # set variations
        elif name == 'epsilon':
            self.epsilon = value
        # set device
        elif name == 'device':
            self.device = value
    

class AApNN(torch.nn.Module):
    def __init__(self, topology, aging_generator, args):
        super().__init__()
        self.args = args
        self.M = args.M_train
        self.K = args.K_train
        if args.MODE == 'nominal':
            self.t = torch.tensor([0.])
        else:
            self.t = torch.linspace(0, 1, self.K)
        self.N = args.N_train
        self.epsilon = args.e_train
        self.model = torch.nn.Sequential()
        self.device = args.DEVICE
        for i in range(len(topology)-1):
            self.model.add_module(f'{i}-th pLayer', AApLayer(topology[i], topology[i+1], aging_generator, args))
    
    def forward(self, X):
        X_extend = X.repeat(self.M, self.K, self.N, 1, 1)
        return self.model(X_extend)
    
    def SetParameter(self, name, value):
        # set time sampling and update K
        if name == 't':
            self.t = value
            self.K = self.t.shape[0]
            for m in self.model:
                m.SetParameter('t', self.t)
        # set number of time sampling K and generate random time sampling
        elif name == 'K':
            self.K = value
            self.t = torch.rand(self.K)
            for m in self.model:
                m.SetParameter('t', self.t)
        # set number of aging-model sampling M
        elif name == 'M':
            self.M = value
            for m in self.model:
                m.SetParameter('M', self.M)
        # set number of samples
        elif name == 'N':
            self.N = value
            for m in self.model:
                m.SetParameter('N', self.N)
        # set variations
        elif name == 'epsilon':
            self.epsilon = value
            for m in self.model:
                m.SetParameter('epsilon', self.epsilon)
        # set device
        elif name == 'device':
            self.device = value
            for m in self.model:
                m.SetParameter('device', self.device)
        

class Lossfunction(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):   
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L
    
    def MonteCarlo(self, prediction, label):
        M = prediction.shape[0]
        K = prediction.shape[1]
        N = prediction.shape[2]
        loss = torch.tensor(0.).to(self.args.DEVICE)
        for m in range(M):
            for k in range(K):
                for n in range(N):
                    loss += self.standard(prediction[m,k,n,:,:], label)
        return loss / M / K / N
    
    def GaussianQuadrature(self, prediction, label):
        return torch.tensor(0.)
    
    def forward(self, prediction, label):
        if self.args.integration == 'MC':
            return self.MonteCarlo(prediction, label)
        elif self.args.integration == 'GQ':
            return self.GaussianQuadrature(prediction, label)