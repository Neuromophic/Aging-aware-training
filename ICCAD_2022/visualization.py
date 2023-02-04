import sys
import os
import pickle
import torch
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')
sys.path.append('../Aging_Model/')
import matplotlib.pyplot as plt
from utils import *
from configuration import *
import FigureConfig as FC

if not os.path.exists('./Figures/'):
    os.makedirs('./Figures/')


for dataset_idx in range(13):
    
    # load dataset
    datasets = os.listdir('../dataset')
    datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
    datasets.sort()
    dataset = datasets[dataset_idx]

    datapath = os.path.join(f'../dataset/{dataset}')
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    data_name  = data['name']

    # result of random guess
    RG = torch.load('../RandomGuess.result')
    Baseline = RG[data_name].item()

    # find files for results
    all_files = os.listdir('./evaluation/')
    target_files = [f for f in all_files if f'{data_name}' in f]
    target_files.sort()
    
    # read results
    result_nominal = []
    result_aging = []

    for seed in range(10):
        target_file_nominal = [f for f in target_files if f'{data_name}_{seed}_nominal' in f][0]
        result_nominal.append(torch.load(f'./evaluation/{target_file_nominal}'))

        target_file_aging = [f for f in target_files if f'{data_name}_{seed}_aging' in f][0]
        result_aging.append(torch.load(f'./evaluation/{target_file_aging}'))

    result_nominal = torch.stack(result_nominal)
    result_aging = torch.stack(result_aging)

    # read the test set according to best validation set
    # for nominal training
    mean_valid_acc_nominal = result_nominal[:,0,:].mean(1)
    best_design_nominal = torch.argmax(mean_valid_acc_nominal)
    final_test_acc_nominal = result_nominal[best_design_nominal, 2, :]
    final_test_std_nominal = result_nominal[best_design_nominal, 3, :]
    # for aging-aware training
    mean_valid_acc_aging = result_aging[:,0,:].mean(1)
    best_design_aging = torch.argmax(mean_valid_acc_aging)
    final_test_acc_aging = result_aging[best_design_aging, 2, :]
    final_test_std_aging = result_aging[best_design_aging, 3, :]

    # print result
    print(f'The mean accuracy on {data_name} dataset from nominal training     w.r.t. the whole device life time is: {final_test_acc_nominal.mean():.3f} $\pm$ {final_test_std_nominal.mean():.3f}')
    print(f'The mean accuracy on {data_name} dataset from aging-aware training w.r.t. the whole device life time is: {final_test_acc_aging.mean():.3f} $\pm$ {final_test_std_aging.mean():.3f}')

    # plot
    t_axis = np.linspace(0,10,500)

    # log10 scale for x-axis
    plt.xscale('log') 

    # draw nominal and aging-aware
    # draw std
    plt.fill_between(t_axis, final_test_acc_nominal-final_test_std_nominal, np.where(final_test_acc_nominal+final_test_std_nominal < 1, final_test_acc_nominal+final_test_std_nominal, 1), alpha=0.3, color=FC.Cyan);
    plt.fill_between(t_axis, final_test_acc_aging-final_test_std_aging, np.where(final_test_acc_aging+final_test_std_aging < 1, final_test_acc_aging+final_test_std_aging, 1), alpha=0.3, color=FC.Pink);
    # draw mean
    plt.plot(t_axis, final_test_acc_nominal, label='PNN', color=FC.Cyan);
    plt.plot(t_axis, final_test_acc_aging, label='AAPNN', color=FC.Pink);
    # draw random guess
    plt.plot(t_axis, np.ones(500)*Baseline, '--', label='Baseline', color=FC.Black);
    # draw train and extrapolation region
    plt.plot(np.ones(500), np.linspace(0,1.2,500), '-', color=FC.Gray, alpha=0.7);

    # draw label and ticks
    plt.title(f'{data_name}', fontsize=25)
    # plt.xlabel('Normalized time $t$', fontsize=20);
    # plt.ylabel('Basic accuracy', fontsize=20);
    plt.xticks(fontsize=20)
    plt.xticks([0.1, 1, 10], ['0.1', '1', '10'])
    plt.yticks(fontsize=20)

    # limite ranges
    plt.xlim([t_axis[0], t_axis[-1]]);
    plt.ylim([max([0, 0.9*min([min(final_test_acc_nominal-final_test_std_nominal), min(final_test_acc_aging-final_test_std_aging)])]),
              min([1.02, 1.05*max([max(final_test_acc_nominal+final_test_std_nominal), max(final_test_acc_aging+final_test_std_aging)])])]);
    # plt.legend();
    plt.savefig(f'./Figures/{data_name}.pdf', format='pdf', bbox_inches='tight')

# draw a legend for all figures once
plt.plot(t_axis, final_test_acc_nominal*0, label='mean of nominal training', color=FC.Cyan);
plt.fill_between(t_axis, final_test_acc_nominal*0, final_test_acc_nominal*0, label='std     of nominal training', alpha=0.3, color=FC.Cyan);
plt.plot(t_axis, final_test_acc_nominal*0, label='mean of aging-aware training', color=FC.Pink);
plt.fill_between(t_axis, final_test_acc_nominal*0, final_test_acc_nominal*0, label='std     of aging-aware training',alpha=0.3, color=FC.Pink);
plt.plot(t_axis, final_test_acc_nominal*0, '--', label='Random guess', color=FC.Black);
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0, 1]);
plt.ylim([0, 1]);
plt.legend();
plt.savefig(f'./Figures/*Legend.pdf', format='pdf', bbox_inches='tight')