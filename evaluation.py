import torch
import PNN_Setting as ps
from tqdm.notebook import tqdm
import numpy as np


def BASIC(nn, x, y, *args, **kwargs):
    ''' no sensing marigin considere ''' 

    act, idx = torch.max(nn(x), dim=1)
    corrects = (y.view(-1) == idx)

    return corrects.float().sum().item()/y.numel()

def MAA(nn, x, y, sensing_margin=.01):
    ''' measure aware accuracy 

        nn : neural network
        x : input x 
        y : target label y 
    '''
        
    # get topk and top activations
    act, idx = torch.topk(nn(x), k=2, dim=1)

    # compare activations top 
    corrects = (act[:, 0] >= sensing_margin) & (act[:, 1] <= 0) & (y.view(-1) == idx[:, 0])
                
    return corrects.float().sum().item()/y.numel()



def FullEvaluation(pnn, valid_loader, M=500, K=50):
    test_time = np.linspace(0, 1, K)
    acc = []
    maa = []

    for omega in tqdm(range(M)):
        pnn.apply(ps.makemodel)
        for test_t in test_time:
            pnn.apply(lambda z: ps.settime(z, test_t))
            for x_valid, y_valid in valid_loader:
                acc.append(BASIC(pnn, x_valid, y_valid))
                maa.append(MAA(pnn, x_valid, y_valid))
        if omega % 10 == 0:
            print(f'evaluating on the {omega}-th model.')
            
    acc = np.array(acc).reshape([M, K])
    mean_acc = np.mean(acc, axis=0).flatten()
    std_acc = np.std(acc, axis=0).flatten()
    
    maa = np.array(maa).reshape([M, K])
    mean_maa = np.mean(maa, axis=0).flatten()
    std_maa = np.std(maa, axis=0).flatten()
    
    return mean_acc, std_acc, mean_maa, std_maa