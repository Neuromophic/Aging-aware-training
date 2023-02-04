import torch
import numpy as np

class Evaluator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sensing_margin = 0.01
        self.performance = self.nominal
        self.args = args
        
    def nominal(self, prediction, label):
        act, idx = torch.max(prediction, dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()
    
    def maa(self, prediction, label):
        act, idx = torch.topk(prediction, k=2, dim=1)
        corrects = (act[:,0] >= self.sensing_margin) & (act[:,1]<=0) & (label.view(-1)==idx[:,0])
        return corrects.float().sum().item() / label.numel()

    def variation(self, nn, x, label, mode='train', scalar=True):
        if mode == 'train' or mode == 'valid':
            prediction = nn(x)
            M, K, N = prediction.shape[0], prediction.shape[1], prediction.shape[2]
        elif mode == 'test':
            M = self.args.M_test
            K = self.args.K_test
            tmax = self.args.t_test_max
            t = torch.linspace(0, tmax, K)
            N = self.args.N_test
            
            nn.SetParameter('M', M)
            nn.SetParameter('t', t)
            nn.SetParameter('N', N)
            
            prediction = nn(x)
            
        accs = torch.zeros(M, K, N)
        for m in range(M):
            for k in range(K):
                for n in range(N):
                    accs[m,k,n] = self.performance(prediction[m,k,n,:,:], label)
        if scalar:
            return torch.mean(accs)
        else:
            return accs
        
    def forward(self, nn, x, label, mode='train', metric='acc', scalar=True):
        if metric == 'acc':
            self.performance = self.nominal
        elif metric == 'maa':
            self.performance = self.maa
        return self.variation(nn, x, label, mode, scalar)
    
  