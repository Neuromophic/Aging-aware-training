import copy
import pNN_aging_aware_vectorization as pnnv
import torch
from torch.autograd import Variable
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pickle


def train_normal_nn(nn, train_loader, valid_loader, optimizer, lossfunction, Epoch=500):
    '''
    trainig of normal NN from PyTorch-default
    :param nn: NN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs
    :return: training loss, validation loss, best parameter
    '''
    # initialization
    best_parameter = copy.deepcopy(nn.state_dict())
    train_loss = []
    valid_loss = []
    averager = [100000]
    best_valid_loss = 100000

    # training
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.cpu().data)

        # validation
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                y_hat = torch.argmax(prediction_valid, 1).cpu().data.numpy().squeeze()
                acc_valid = sum(y_hat == y_valid.cpu().numpy()) / y_valid.shape[0]
                loss_valid = lossfunction(prediction_valid, y_valid)
        # save best parameter
        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            best_parameter = copy.deepcopy(nn.state_dict())
        valid_loss.append(loss_valid.cpu().data.item())
        
        
        if not epoch % 500:
                print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid.data:.9f}')
                
        if epoch >= Epoch*0.2:
            if not epoch % 100:
                averager.append(np.mean(valid_loss[-5000::100] * np.linspace(0,1,50)))
            if averager[-2] <= averager[-1]:
                print('Early stop.')
                break
                
    print('Finished.')
    return train_loss, valid_loss, best_parameter



def train_normal_pnn(nn, train_loader, valid_loader, m, T, optimizer, lossfunction, Epoch=500, cache='default'):
    '''
    nominal training for pNN
    :param nn: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs
    :param cache: path to save cache data
    :return: 
    '''
    # initialization
    best_parameter = copy.deepcopy(nn.state_dict())
    train_loss = []
    valid_loss = []
    averager = [100000]
    best_valid_loss = 100000

    # training
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            
            xv_train = x_train.repeat(1,1,1,1)
            prediction = nn(xv_train)
            
            loss = lossfunction(prediction, y_train)
            loss.backward()
            optimizer.step()
            
        train_loss.append(copy.deepcopy(loss.cpu().data.item()))

        # validation
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                xv_valid = x_valid.repeat(1,1,1,1)
                prediction_valid = nn(xv_valid)

                loss_valid = lossfunction(prediction_valid, y_valid)

                y_hat = torch.argmax(prediction_valid, 3).cpu().flatten().data.numpy().squeeze()
                y_valid = y_valid.repeat(1,1,1).cpu().flatten().data.numpy().squeeze()
                acc_valid = sum(y_hat == y_valid) / y_valid.shape[0]

        # save best parameter
        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            best_parameter = copy.deepcopy(nn.state_dict())
            # save cache
            with open(f'./temp/{cache}_PNN.p', 'wb') as f:
                pickle.dump(nn, f)
                
        valid_loss.append(copy.deepcopy(loss_valid.cpu().data.item()))
        
        if not epoch % 500:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid.data:.9f} |')
        
        if epoch >= Epoch*0.4:
            if not epoch % 100:
                averager.append(np.mean(valid_loss[-5000::100] * np.linspace(0,1,50)))
            if averager[-2] <= averager[-1]:
                print('Early stop.')
                break
        
    print('Finished.')
    return train_loss, valid_loss, best_parameter


def train_aged_pnn(nn, train_loader, valid_loader, m, T, M, K, M_valid, K_valid, optimizer, lossfunction, Epoch=500, cache='default'):
    '''
    aging aware training for pNN
    :param nn: pNN
    :param train_loader: data loader for training data
    :param valid_loader: data loader for validation data
    :param m: sensing margin
    :param T: sensing related hyperparameter
    :param M: number of aging models in training
    :param K: number of timings in training
    :param M_valid: number of aging models in validation
    :param K_valid: number of timings in validation
    :param optimizer: optimizer
    :param lossfunction: loss function
    :param Epoch: max. epochs
    :param cache: path for saving cache data
    :return: training loss, validation loss, best parameters
    '''
    best_parameter = copy.deepcopy(nn.state_dict())
    valid_loss = []
    train_loss = []
    best_valid_loss = 1000
    
    for epoch in tqdm(range(Epoch)):

        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            
            nn.apply(lambda z: pnnv.MakeModel(z, M))
            nn.apply(lambda z: pnnv.SetTime(z, np.random.rand(K).tolist()))
            
            xv_train = x_train.repeat(M,K,1,1)
            prediction = nn(xv_train)

            loss = lossfunction(prediction, y_train)
            loss.backward()

            optimizer.step()  
            
        train_loss.append(loss.cpu().data.item())
        
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                nn.apply(lambda z: pnnv.MakeModel(z, M=M_valid))
                nn.apply(lambda z: pnnv.SetTime(z, np.random.rand(K_valid).tolist()))

                xv_valid = x_valid.repeat(M_valid,K_valid,1,1)
                prediction_valid = nn(xv_valid)

                loss_valid = lossfunction(prediction_valid, y_valid)

                y_hat = torch.argmax(prediction_valid, 3).flatten().cpu().data.numpy().squeeze()
                y_valid = y_valid.repeat(M_valid,K_valid,1).flatten().cpu().data.numpy().squeeze()
                acc_valid = sum(y_hat == y_valid) / y_valid.shape[0]
        
        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            best_parameter = copy.deepcopy(nn.state_dict())
            
            with open(f'./temp/{cache}_AAPNN.p', 'wb') as f:
                pickle.dump(nn, f)
            
        valid_loss.append(loss_valid.cpu().data.item())
        
        if not epoch % int(Epoch/20):
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid:.9f} |')
            
    print('Finished.')
    return train_loss, valid_loss, best_parameter

        