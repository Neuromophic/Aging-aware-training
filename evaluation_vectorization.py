import torch
import pNN_aging_aware_vectorization as pnnv
from tqdm.notebook import tqdm
import numpy as np

def ACC(prediction, target, device):
    y_hat = torch.argmax(prediction, 3)
    
    corrects = (y_hat == target).int()
    
    num_corrects = torch.sum(corrects, dim=2, keepdim=True)
    acc = num_corrects / torch.tensor(target.shape[2]).to(device)
    
    del y_hat, corrects, num_corrects
    torch.cuda.empty_cache()
    
    return acc.squeeze().cpu().numpy()

def MAA(prediction, target, device, sensing_margin=.01):
    act, idx = torch.topk(prediction, k=2, dim=3)
    
    corrects = ((act[:,:,:, 0] >= 0.03) & (act[:,:,:, 1] <= 0) & (target == idx[:,:,:, 0])).int()
    
    num_corrects = torch.sum(corrects, dim=2, keepdim=True)
    acc = num_corrects / torch.tensor(target.shape[2])
    
    del act, idx, corrects, num_corrects
    torch.cuda.empty_cache()
    
    return acc.squeeze().cpu().numpy()

def Evaluation(nn, valid_loader, M_valid, M_max, K_valid, device):
    accs = []
    maas = []
    
    nn.apply(lambda z: pnnv.SetDevice(z, device))
    nn.apply(lambda z: pnnv.SetTime(z, np.linspace(0,1,K_valid).tolist()))
    
    
    Iteration = int(M_valid / M_max)
    
    for iteration in tqdm(range(Iteration)):
        nn.apply(lambda z: pnnv.MakeModel(z, M_max))
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                xv_valid = x_valid.repeat(M_max, K_valid, 1, 1)
                prediction = nn(xv_valid)

                yv_valid = y_valid.repeat(M_max, K_valid, 1)

                acc = ACC(prediction, yv_valid, device)
                maa = MAA(prediction, yv_valid, device)
                
                accs.append(acc)
                maas.append(maa)
    
    acc = np.array(accs).reshape(M_valid, K_valid)
    maa = np.array(maas).reshape(M_valid, K_valid)
    
    mean_acc = np.mean(acc, axis=0).flatten()
    std_acc = np.std(acc, axis=0).flatten()
    
    mean_maa = np.mean(maa, axis=0).flatten()
    std_maa = np.std(maa, axis=0).flatten()
    
    return mean_acc, std_acc, mean_maa, std_maa

def GetStructure(nn):
    neurons = []

    if isinstance(nn[0], torch.nn.Linear):
        for l in nn:
            if isinstance(l, torch.nn.Linear):
                neurons.append(l.in_features)
                neurons.append(l.out_features)
    else:
        for l in nn:
            for p in l.parameters():
                neurons.append(p.shape[1]-2)
                neurons.append(p.shape[0])
    

    neurons.append(neurons[-1])
    neurons = neurons[::2]
    
    structure = ''
    for neuron in neurons:
        structure += '_'
        structure += str(neuron)
        
    return structure
        