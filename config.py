import os

path = os.path.join(os.getcwd(), 'Datasets', 'dataset_processed')
files = os.listdir(path)
datasets = [k.replace('Dataset_','').replace('.p','') for k in files if k.startswith('Dataset_')]
datasets.sort()

current_dataset = 12
