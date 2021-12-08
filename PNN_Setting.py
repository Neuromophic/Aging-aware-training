import numpy as np
import torch
import copy
import pNN_aging_aware as pnn

def set_time(pnn, t):
    for l in pnn:
        l.t = t
    return pnn

def make_model(pnn):
    for l in pnn:
        l.generate_aging_model()
    return pnn

def settime(m, t):
    if isinstance(m, pnn.PNNLayer):
        m.t = t
        
def makemodel(m):
    if isinstance(m, pnn.PNNLayer):
        m.generate_aging_model()
        
def zerogradient(m):
    if isinstance(m, pnn.PNNLayer):
        for p in m.parameters():
            if p.grad is not None:
                p.grad = torch.zeros_like(p.grad)

def MakeParallelPNNs(pnn, M, K):
    Parallel_PNNs = []
    # copy M PNNs
    pnn_with_different_models = [make_model(copy.deepcopy(pnn)) for m in range(M)]
    # for each copyed PNN
    for n in pnn_with_different_models:        
        # copy each aging model K times
        pnn_with_different_times = [copy.deepcopy(set_time(n, np.random.rand())) for k in range(K)]
        # give them different time stamps
        Parallel_PNNs.append(pnn_with_different_times)
    # convert 2d list to 1d
    Parallel_PNNs = [item for sublist in Parallel_PNNs for item in sublist]
    return Parallel_PNNs

def MakeParallelModels(pnn, M):
    # copy M PNNs
    pnn_with_different_models = [copy.deepcopy(pnn) for m in range(M)]
    # change aging model for each pnn
    for n in pnn_with_different_models:
        n.apply(makemodel)
    return pnn_with_different_models