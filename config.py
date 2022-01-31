import os

path = os.path.join(os.getcwd(), 'Datasets', 'dataset_processed')
files = os.listdir(path)
datasets = [k.replace('Dataset_','').replace('.p','') for k in files if k.startswith('Dataset_')]
datasets.sort()

current_dataset = 11

lr = 0.1

AAPNN_lr_1 = 0.05
AAPNN_lr_2 = 0.05
AAPNN_lr_3 = 0.01

data_split_seed = 0

N_Hidden = 3

m = 0.3
T = 0.1

M_train = 50
K_train = 10
M_test = 20
K_test = 10
M_valid = 500
K_valid = 50