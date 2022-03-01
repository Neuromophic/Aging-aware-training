import torch
import PNN_Setting as ps
from tqdm.notebook import tqdm
import numpy as np


def BASIC(nn, x, y, *args, **kwargs):
    '''
    classic accuracy of NN
    :param nn: neural network
    :param x: testing examples
    :param y: testing labels
    :return: accuracy
    '''
    act, idx = torch.max(nn(x), dim=1)
    corrects = (y.view(-1) == idx)

    return corrects.float().sum().item() / y.numel()


def MAA(nn, x, y, sensing_margin=.01):
    '''
    measuring-aware accuracy, considering the measuring-margin
    and the output of wrong classes, i.e., the output of wrong
    classes must be negative, otherwise is the classification still wrong.
    :param nn: neural network
    :param x: testing examples
    :param y: testing labels
    :param sensing_margin: sensing margin
    :return: measuring-aware accuracy
    '''
    # find the 2 highest output in predictions
    act, idx = torch.topk(nn(x), k=2, dim=1)
    # if the class with the highest output is larger than sensing-margin (sensible)
    # and the 2nd highest value is negative
    # the of course the highest output refers to the right class
    # then the classification is right
    corrects = (act[:, 0] >= sensing_margin) & (act[:, 1] <= 0) & (y.view(-1) == idx[:, 0])

    return corrects.float().sum().item() / y.numel()


def FullEvaluation(pnn, test_loader, M=500, K=50):
    '''
    perform testing multiple times, w.r.t. different aging-models and timings.
    both MAA and ACC will be calculated.
    the mean and std in terms of aging-models will also be calculated
    :param pnn: model
    :param test_loader: test dataset
    :param M: number of sampled aging-models
    :param K: number of sampled timings
    :return: mean and std of classic accuracy and measuring-aware accuracy w.r.t. aging-models over time
    '''
    # initialize lists for maa and acc
    acc = []
    maa = []
    # sampling timings
    test_time = np.linspace(0, 1, K)

    # loop for aging models
    for omega in tqdm(range(M)):
        # generate aging models
        pnn.apply(ps.makemodel)
        # loop for timing
        for test_t in test_time:
            # calculate $A_{omega}(t)$
            pnn.apply(lambda z: ps.settime(z, test_t))
            # calculate acc and maa
            for x_test, y_test in test_loader:
                acc.append(BASIC(pnn, x_test, y_test))
                maa.append(MAA(pnn, x_test, y_test))
        if omega % 10 == 0:
            print(f'evaluating on the {omega}-th model.')

    # calculate mean and std for acc
    acc = np.array(acc).reshape([M, K])
    mean_acc = np.mean(acc, axis=0).flatten()
    std_acc = np.std(acc, axis=0).flatten()

    # calculate mean and std for maa
    maa = np.array(maa).reshape([M, K])
    mean_maa = np.mean(maa, axis=0).flatten()
    std_maa = np.std(maa, axis=0).flatten()

    return mean_acc, std_acc, mean_maa, std_maa
