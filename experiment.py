#!/usr/bin/env python

#SBATCH --job-name=Aging

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'Aging_Model'))
import torch
import pickle
import os
from configuration import *
from torch.utils.data import TensorDataset, DataLoader
import pNN_aging as pNN
from utils import *

args = parser.parse_args()

with open('./Aging_Model/exp_aging_model.p', 'rb') as f:
    age_generator = pickle.load(f)

args.DEVICE = CheckDevice(args)
print(f'Training network on device: {args.DEVICE}.')

MakeFolder(args)

if not args.VARIATION:
    args.e_train = 0.
    args.N_train = 1
if args.MODE=='nominal':
    args.M_train = 1
    args.K_train = 1

datasets = os.listdir('./dataset')
datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
datasets.sort()

dataset = datasets[args.DATASET]
datapath = os.path.join(f'./dataset/{dataset}')
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

# generate tensordataset
trainset = TensorDataset(X_train, y_train)
validset = TensorDataset(X_valid, y_valid)
testset  = TensorDataset(X_test, y_test)
# batch
train_loader = DataLoader(trainset, batch_size=N_train)
valid_loader = DataLoader(validset, batch_size=N_valid)
test_loader  = DataLoader(testset,  batch_size=N_test)

SetSeed(args.SEED)

setup = f'{data_name}_{args.SEED}_{args.MODE}_{args.e_train}_{args.N_train}_{args.integration}'
print(f'Training setup: {setup}.')

msglogger = GetMessageLogger(args, setup)
msglogger.info(f'Training network on device: {args.DEVICE}.')
msglogger.info(f'Training setup: {setup}.')
msglogger.info(f'Dataset "{data_name}" has {N_feature} input features and {N_class} classes. There are {N_train} training examples, {N_valid} valid examples, and {N_test} test examples in the dataset.')


if os.path.isfile(f'{args.savepath}/pNN_{setup}'):
    print(f'{setup} exists, skip this training.')
    msglogger.info('Training was already finished.')
else:
    topology = [N_feature] + args.hidden + [N_class]
    msglogger.info(f'Topology of the network: {topology}.')
    
    aapnn = pNN.AApNN(topology, age_generator, args).to(args.DEVICE)

    lossfunction = pNN.Lossfunction(args).to(args.DEVICE)
    optimizer = torch.optim.Adam(aapnn.parameters(), lr=args.LR)

    if args.PROGRESSIVE:
        aapnn, best = train_pnn_with_patience(aapnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        aapnn, best = train_pnn(aapnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    
    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(aapnn.to('cpu'), f'{args.savepath}/pNN_{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')