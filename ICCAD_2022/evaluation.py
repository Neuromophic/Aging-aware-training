#!/usr/bin/env python

#SBATCH --job-name=AgingEva

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:4

import sys
import os
import pickle
import torch
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')
sys.path.append('../Aging_Model/')
from utils import *
from configuration import *

if not os.path.exists('./evaluation/'):
    os.makedirs('./evaluation/')
    
    
args = parser.parse_args()
args.N_test = 1
args.e_test = 0.
args.DEVICE = CheckDevice(args)

datasets = os.listdir('../dataset')
datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
datasets.sort()

dataset = datasets[args.DATASET]

datapath = os.path.join(f'../dataset/{dataset}')
with open(datapath, 'rb') as f:
    data = pickle.load(f)
    
X_train    = data['X_train'].to(args.DEVICE)
y_train    = data['y_train'].to(args.DEVICE)
X_valid    = data['X_valid'].to(args.DEVICE)
y_valid    = data['y_valid'].to(args.DEVICE)
X_test     = data['X_test'].to(args.DEVICE)
y_test     = data['y_test'].to(args.DEVICE)
data_name  = data['name']

N_class    = data['n_class']
N_feature  = data['n_feature']
N_train    = X_train.shape[0]
N_valid    = X_valid.shape[0]
N_test     = X_test.shape[0]

print(f'Dataset "{data_name}" has {N_feature} input features and {N_class} classes.\nThere are {N_train} training examples, {N_valid} valid examples, and {N_test} test examples in the dataset.')

modelname = f'pNN_{data_name}_{args.SEED}_{args.MODE}_0.0_1_MC'

model = torch.load(f'./models/{modelname}').to(args.DEVICE)
model.SetParameter('device', args.DEVICE)

SetSeed(args.SEED)
evaluator = Evaluator(args).to(args.DEVICE)

acc_valid = evaluator(model, X_valid, y_valid, mode='test', metric='maa', scalar=False)
acc_test = evaluator(model, X_test,  y_test,  mode='test', metric='maa', scalar=False)

ACC_valid = torch.mean(acc_valid, dim=[0,2])
STD_valid = torch.std(acc_valid,  dim=[0,2])
ACC_test  = torch.mean(acc_test,  dim=[0,2])
STD_test  = torch.std(acc_test,   dim=[0,2])
results = torch.stack([ACC_valid, STD_valid, ACC_test, STD_test], dim=0)

torch.save(results, f'./evaluation/result_{data_name}_{args.SEED}_{args.MODE}_0.0_1_MC')