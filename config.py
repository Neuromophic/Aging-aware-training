import os

# path for dataset
path = os.path.join(os.getcwd(), 'Datasets', 'dataset_processed')
files = os.listdir(path)
datasets = [k.replace('Dataset_', '').replace('.p', '') for k in files if k.startswith('Dataset_')]
datasets.sort()

# which dataset is selected
current_dataset = 8

# learning-rate
lr = 0.1

# random seed for data-split
data_split_seed = 0

# architecture for pNN
N_Hidden = 3

# measuring-aware hyperparameter
m = 0.3
T = 0.1

# training-related hyperparameter
M_train = 50
K_train = 10
M_valid = 20
K_valid = 10
M_test = 500
K_test = 50

# extension to topology
Topology = 12