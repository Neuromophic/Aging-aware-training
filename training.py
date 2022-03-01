import copy
import pNN_aging_aware as pnn
import torch
from torch.autograd import Variable
import numpy as np
import PNN_Setting as ps
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pickle


def train_normal_nn(nn, train_loader, valid_loader, optimizer, lossfunction, Epoch=500):
    '''
    train normal NN build from PyTorch-default
    :param nn: NN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for valid data
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs for training
    :return: training loss, validation loss, parameters with lowest validation loss
    '''
    # initialization
    best_parameter = copy.deepcopy(nn.state_dict())
    train_loss = []
    valid_loss = []
    best_valid_loss = 100000

    # training
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.data)

        # validation
        for x_valid, y_valid in valid_loader:
            prediction_valid = nn(x_valid)
            y_hat = torch.argmax(prediction_valid, 1).data.numpy().squeeze()
            acc_valid = sum(y_hat == y_valid.numpy()) / y_valid.shape[0]
            loss_valid = lossfunction(prediction_valid, y_valid)

        # save best parameters
        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            best_parameter = copy.deepcopy(nn.state_dict())
        valid_loss.append(loss_valid.data)

        if not epoch % 100:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid.data:.9f} |')

    print('Finished.')
    return train_loss, valid_loss, best_parameter


def train_normal_pnn(nn, train_loader, valid_loader, m, T, optimizer, lossfunction, Epoch=500):
    '''
    nominal training for pNN
    :param nn: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs for training
    :return: training loss, validation loss and parameters with lowest validation loss
    '''
    # initialization
    best_parameter = copy.deepcopy(nn.state_dict())
    train_loss = []
    valid_loss = []
    best_valid_loss = 10**10

    # training
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train, m, T)
            loss.backward()
            optimizer.step()
        train_loss.append(copy.deepcopy(loss.data))

        # validation
        for x_valid, y_valid in valid_loader:
            prediction_valid = nn(x_valid)
            y_hat = torch.argmax(prediction_valid, 1).data.numpy().squeeze()
            acc_valid = sum(y_hat == y_valid.numpy()) / y_valid.shape[0]
            loss_valid = lossfunction(prediction_valid, y_valid, m, T)

        # save best parameters and save training process in /temp/ file
        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            best_parameter = copy.deepcopy(nn.state_dict())

            with open('./temp/PNN.p', 'wb') as f:
                pickle.dump(nn, f)

        valid_loss.append(copy.deepcopy(loss_valid.data))

        if not epoch % 100:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid.data:.9f} |')

    print('Finished.')
    return train_loss, valid_loss, best_parameter


def train_aged_pnn(nn, train_loader, valid_loader, m, T, M, K, M_valid, K_valid, optimizer, lossfunction, Epoch=500):
    '''
    aging aware training for pnn
    :param nn: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param M: number of aging models while training
    :param K: number of timings while training
    :param M_valid: number of aging models while validation
    :param K_valid: number of timings while validation
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs for training
    :return: validation loss and best parameter
    '''
    # initialization
    best_parameter = copy.deepcopy(nn.state_dict())
    valid_loss = []
    best_valid_loss = 1000

    # training
    for epoch in tqdm(range(Epoch)):

        optimizer.zero_grad()
        # generate aging models
        for omega in range(M):
            nn.apply(ps.makemodel)
            # generate timings
            for k in range(K):
                nn.apply(lambda z: ps.settime(z, np.random.rand()))
                # training
                for X_train, y_train in train_loader:
                    prediction = nn(X_train)

                    loss = lossfunction(prediction, y_train, m, T)
                    loss.backward()

        optimizer.step()

        # validation
        avg_loss, avg_acc = Valid(nn, lossfunction, m, T, valid_loader, M_valid, K_valid)

        # averager of previous validation loss
        # save best parameters
        if avg_loss.data < best_valid_loss:
            best_valid_loss = avg_loss.data
            best_parameter = copy.deepcopy(nn.state_dict())
        valid_loss.append(avg_loss.data)

        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.9f} |')

    print('Finished.')
    return valid_loss, best_parameter


def train_aged_pnn_parallel(nn, train_loader, valid_loader, m, T, M, K, M_valid, K_valid, optimizer, lossfunction,
                            Epoch=500):
    '''
    parallel aging-aware training with multiple cpus
    :param nn: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param M: number of aging models while training
    :param K: number of timings while training
    :param M_valid: number of aging models while validation
    :param K_valid: number of timings while validation
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs for training
    :return: validation loss and best parameter
    '''
    best_parameter = copy.deepcopy(nn.state_dict())
    valid_loss = []
    best_valid_loss = 1000

    for epoch in tqdm(range(Epoch)):
        # clear gradient of AAPNN
        optimizer.zero_grad()

        # copy aapnn M times for parallel training
        Parallel_Models = ps.MakeParallelModels(nn, M)

        # for each aapnn
        for aapnn in Parallel_Models:
            # clear gradient
            aapnn.apply(ps.zerogradient)
            # sample K time stamps
            for k in range(K):
                aapnn.apply(lambda z: ps.settime(z, np.random.rand()))
                # apply forward propagation
                for X_train, y_train in train_loader:
                    prediction = aapnn(X_train)

                    # calculate loss and do back propagation
                    loss = lossfunction(prediction, y_train, m, T)
                    loss.backward()

        # get gradients for each layer of AAPNN
        for n, p in nn.named_parameters():
            # enlarge 1 dim for torch.cat(), i.e. temp is [M, n_out, n_in+2] dimensional tensor
            temp = torch.cat([dict(pnn_temp.named_parameters())[n].grad[None, :]
                              for pnn_temp in Parallel_Models])
            # average w.r.t. 0. dimension, i.e. M parallel aapnns
            # devide K to average w.r.t. K time samples
            p.grad = torch.mean(temp, dim=0) / K

            # update parameter
        optimizer.step()

        # validation
        avg_loss, avg_acc = Valid(nn, lossfunction, m, T, valid_loader, M_valid, K_valid)

        # averager of previous validation loss
        # save best parameters
        if avg_loss.data < best_valid_loss:
            best_valid_loss = avg_loss.data
            best_parameter = copy.deepcopy(nn.state_dict())
        valid_loss.append(avg_loss.data)

        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.9f} |')

    print('Finished.')
    return valid_loss, best_parameter


def GetGradientFor(aapnn, train_loader, m, T, K, lossfunction):
    '''
    calculate gradient for parallel training
    :param aapnn: pNN
    :param train_loader: data loader for training data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param K: number of timings in training
    :param lossfunction: loss function
    :return: dictionary of gradients
    '''
    aapnn.apply(ps.zerogradient)
    # sample K time stamps
    for k in range(K):
        aapnn.apply(lambda z: ps.settime(z, np.random.rand()))
        # apply forward propagation
        for X_train, y_train in train_loader:
            prediction = aapnn(X_train)

            # calculate loss and do back propagation
            loss = lossfunction(prediction, y_train, m, T)
            loss.backward()

    grad_dict = {}
    for n, p in aapnn.named_parameters():
        grad_dict[n] = p.grad

    return grad_dict


def UpDateParameter(aapnn, gradients, K, optimizer):
    '''
    update parameters in pNN
    :param aapnn: pNN
    :param gradients: gradients of parameters
    :param K: number of timings
    :param optimizer: optimizer
    :return: updated pNN
    '''
    optimizer.zero_grad()
    # calculate the mean gradient of K parallel pNNs
    for n, p in aapnn.named_parameters():
        temp = torch.cat([gradient[n][None, :] for gradient in gradients])
        p.grad = torch.mean(temp, dim=0) / K
    optimizer.step()
    return aapnn


def Valid(nn, lossfunction, m, T, valid_loader, M, K):
    '''
    validation for pNN
    :param nn: pNN
    :param lossfunction: loss function
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param valid_loader: data loader for validation
    :param M: number of aging models in validation
    :param K: number of timings in validation
    :return: loss and accuracy
    '''
    avg_loss = 0
    avg_acc = 0
    valid_time = np.linspace(0, 1, K)
    for omega in range(M):
        nn.apply(ps.makemodel)
        for valid_t in valid_time:
            nn.apply(lambda z: ps.settime(z, valid_t))
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                loss_valid = lossfunction(prediction_valid, y_valid, m, T)
                y_hat = torch.argmax(prediction_valid, dim=1).data.numpy().squeeze()
                acc_valid = sum(y_hat == y_valid.numpy()) / y_valid.shape[0]

                avg_loss += loss_valid.data
                avg_acc += acc_valid

    avg_loss /= (M * K)
    avg_acc /= (M * K)
    return avg_loss, avg_acc


def ParallelTrainingAAPNN(AAPNN, train_loader, valid_loader, optimizer, lossfunction, m, T, M, K, M_valid, K_valid, Epoch):
    '''
    parallel aging-aware training
    :param AAPNN: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param M: number of aging models while training
    :param K: number of timings while training
    :param M_valid: number of aging models while validation
    :param K_valid: number of timings while validation
    :param Epoch: max. epochs of training
    :return: validation loss and best parameters
    '''
    # initialization
    valid_loss = []
    best_loss = 10000
    parameter = copy.deepcopy(AAPNN.state_dict())

    for epoch in tqdm(range(Epoch)):
        # copy pNNs for parallel training
        Parallel_Models = ps.MakeParallelModels(AAPNN, M)
        # calculate gradients
        gradients = Parallel(n_jobs=M)(delayed(GetGradientFor)(aapnn,
                                                               train_loader,
                                                               m, T, K,
                                                               lossfunction) for aapnn in Parallel_Models)
        # update parameters in pNN
        UpDateParameter(AAPNN, gradients, K, optimizer)

        # validation
        loss_temp, acc_temp = Valid(AAPNN, lossfunction, m, T, valid_loader, M_valid, K_valid)

        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_temp:.5f} | Loss: {loss_temp:.9f} |')

        # save best parameter
        valid_loss.append(loss_temp)
        if loss_temp < best_loss:
            best_loss = loss_temp
            parameter = copy.deepcopy(AAPNN.state_dict())

            with open('./temp/AAPNN.p', 'wb') as f:
                pickle.dump(AAPNN, f)

    return valid_loss, parameter
