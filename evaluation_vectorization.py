import torch
import pNN_aging_aware_vectorization as pnnv
from tqdm.notebook import tqdm
import numpy as np
import config


def ACC(prediction, target, device):
    '''
    calculate the classic accuracy
    :param prediction: prediction from NN
    :param target: label
    :param device: cpu/gpu
    :return: classic accuracy
    '''
    # find the predicted class
    y_hat = torch.argmax(prediction, 3)
    # check the right classifications
    corrects = (y_hat == target).int()
    num_corrects = torch.sum(corrects, dim=2, keepdim=True)
    # calculate acc
    acc = num_corrects / torch.tensor(target.shape[2]).to(device)

    # for device memory
    del y_hat, corrects, num_corrects
    torch.cuda.empty_cache()

    return acc.squeeze().cpu().numpy()


def MAA(prediction, target, device, sensing_margin=.01):
    '''
    calculate the measuring-aware accuracy
    :param prediction: prediction from NN
    :param target: label
    :param device: cpu/gpu
    :param sensing_margin: hardware-related parameter
    :return: maa
    '''
    # find the 2 highest output in predictions
    act, idx = torch.topk(prediction, k=2, dim=3)
    # if the class with the highest output is larger than sensing-margin (sensible)
    # and the 2nd highest value is negative
    # the of course the highest output refers to the right class
    # then the classification is right
    corrects = ((act[:, :, :, 0] >= 0.03) & (act[:, :, :, 1] <= 0) & (target == idx[:, :, :, 0])).int()
    # calculate corrects and acc
    num_corrects = torch.sum(corrects, dim=2, keepdim=True)
    acc = num_corrects / torch.tensor(target.shape[2])

    # for device memory
    del act, idx, corrects, num_corrects
    torch.cuda.empty_cache()

    return acc.squeeze().cpu().numpy()


def Evaluation(nn, test_loader, M_test, M_max, K_test, device):
    '''
    calculate maa and acc
    :param nn: neural network
    :param test_loader: test dataset
    :param M_test: number of aging-models in test
    :param M_max: to avoid high memory usage, each time will only a number of aging models tested
    :param K_test: number of timings in test
    :param device: cpu/gpu
    :return: mean and std of maa and acc
    '''
    # initialization for acc and maa
    accs = []
    maas = []
    losses = []

    # set device and timing
    nn.apply(lambda z: pnnv.SetDevice(z, device))
    nn.apply(lambda z: pnnv.SetTime(z, np.linspace(0, 1, K_test).tolist()))

    # how many iteration do we need to reach M_test
    Iteration = int(M_test / M_max)

    # evaluation
    for iteration in tqdm(range(Iteration)):
        # generate aging models
        nn.apply(lambda z: pnnv.MakeModel(z, M_max))
        with torch.no_grad():
            for x_test, y_test in test_loader:
                # vectorization of testing example
                xv_test = x_test.repeat(M_max, K_test, 1, 1)
                # inference
                prediction = nn(xv_test)
                # vectorization of testing labels
                yv_test = y_test.repeat(M_max, K_test, 1)
                
                # save loss
                loss_temp = pnnv.LossFunction(prediction, y_test, dimension=[0,2])[0].item()
                losses.append(loss_temp)
                
                # calculate maa and acc
                acc = ACC(prediction, yv_test, device)
                maa = MAA(prediction, yv_test, device)

                accs.append(acc)
                maas.append(maa)

    # calculate mean and std of maa and acc in terms of aging models
    acc = np.array(accs).reshape(M_test, K_test)
    maa = np.array(maas).reshape(M_test, K_test)

    mean_acc = np.mean(acc, axis=0).flatten()
    std_acc = np.std(acc, axis=0).flatten()

    mean_maa = np.mean(maa, axis=0).flatten()
    std_maa = np.std(maa, axis=0).flatten()
    
    loss = np.mean(losses)
    return mean_acc, std_acc, mean_maa, std_maa, loss


def GetStructure(nn):
    '''
    get the structure of a pNN
    :param nn: pNN
    :return: string to describe the topology of pNN
    '''
    # initialization
    neurons = []

    # get in- and output-neurons
    if isinstance(nn[0], torch.nn.Linear):
        for l in nn:
            if isinstance(l, torch.nn.Linear):
                neurons.append(l.in_features)
                neurons.append(l.out_features)
    else:
        for l in nn:
            for p in l.parameters():
                neurons.append(p.shape[1] - 2)
                neurons.append(p.shape[0])
    # processing
    neurons.append(neurons[-1])
    neurons = neurons[::2]

    # generate string
    structure = ''
    for neuron in neurons:
        structure += '_'
        structure += str(neuron)

    return structure
