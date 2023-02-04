import sys
import torch
import numpy as np
import pickle
import os

datasets = os.listdir('./dataset')
datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
datasets.sort()

RG = {}
for dataset in datasets:
    datapath = os.path.join(f'./dataset/{dataset}')
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    y_train    = data['y_train']
    y_valid    = data['y_valid']
    y_test     = data['y_test']
    N_class    = data['n_class']
    data_name  = data['name']

    y_learn = torch.cat([y_train, y_valid])
    frequence = np.histogram(y_learn, bins=N_class)[0]
    label = np.histogram(y_learn, bins=N_class-1)[1]
    Guess = torch.ones(y_test.numel()) * int(label[np.argmax(frequence)])
    ACC = (y_test==Guess).sum() / y_test.numel()
    
    RG[data_name] = ACC
    
torch.save(RG, 'RandomGuess.result')