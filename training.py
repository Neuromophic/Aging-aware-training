import copy
import pNN_aging_aware as pnn
import torch
from torch.autograd import Variable
import numpy as np
import PNN_Setting as ps
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pickle

def train_normal_nn(nn, train_loader, test_loader, optimizer, lossfunction, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytrainloss = []
    mytestloss = []
    best_test_loss = 100000
    
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train)
            loss.backward()
            optimizer.step()
        mytrainloss.append(loss.data)
        
        for x_test, y_test in test_loader:
            prediction_test = nn(x_test)
            y_hat = torch.argmax(prediction_test, 1).data.numpy().squeeze()
            acc_test = sum(y_hat == y_test.numpy()) / y_test.shape[0]
            loss_test = lossfunction(prediction_test, y_test)

        if loss_test.data < best_test_loss:
            best_test_loss = loss_test.data
            myparameter = copy.deepcopy(nn.state_dict())
        mytestloss.append(loss_test.data)
        
        if not epoch % 100:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_test:.5f} | Loss: {loss_test.data:.9f} |')
        
    print('Finished.')
    return mytrainloss, mytestloss, myparameter

def train_normal_pnn(nn, train_loader, test_loader, m, T, optimizer, lossfunction, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytrainloss = []
    mytestloss = []
    best_test_loss = 100000
    
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train, m, T)
            loss.backward()
            optimizer.step()
        mytrainloss.append(copy.deepcopy(loss.data))
        
        for x_test, y_test in test_loader:
            prediction_test = nn(x_test)
            y_hat = torch.argmax(prediction_test, 1).data.numpy().squeeze()
            acc_test = sum(y_hat == y_test.numpy()) / y_test.shape[0]
            loss_test = lossfunction(prediction_test, y_test, m, T)

        if loss_test.data < best_test_loss:
            best_test_loss = loss_test.data
            myparameter = copy.deepcopy(nn.state_dict())
            
            with open('./temp/PNN.p', 'wb') as f:
                pickle.dump(nn, f)
                
        mytestloss.append(copy.deepcopy(loss_test.data))
        
        if not epoch % 100:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_test:.5f} | Loss: {loss_test.data:.9f} |')
        
    print('Finished.')
    return mytrainloss, mytestloss, myparameter

def train_aged_pnn(nn, train_loader, test_loader, m, T, M, K, M_test, K_test, optimizer, lossfunction, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytestloss = []
    best_test_loss = 1000
    
    for epoch in tqdm(range(Epoch)):

        optimizer.zero_grad()

        for omega in range(M):
            nn.apply(ps.makemodel)            
            for k in range(K):
                nn.apply(lambda z: ps.settime(z, np.random.rand()))

                for X_train, y_train in train_loader:
                    prediction = nn(X_train)

                    loss = lossfunction(prediction, y_train, m, T)
                    loss.backward()

        optimizer.step()   
    
        avg_loss, avg_acc = Test(nn, lossfunction, m, T, test_loader, M_test, K_test)
        
        if avg_loss.data < best_test_loss:
            best_test_loss = avg_loss.data
            myparameter = copy.deepcopy(nn.state_dict())
        mytestloss.append(avg_loss.data)
        
        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.9f} |')
            
    print('Finished.')
    return myloss, myparameter

def train_aged_pnn_parallel(nn, train_loader, test_loader, m, T, M, K, M_test, K_test, optimizer, lossfunction, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytestloss = []
    best_test_loss = 1000
    
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
    
        avg_loss, avg_acc = Test(nn, lossfunction, m, T, test_loader, M_test, K_test)
        
        
        if avg_loss.data < best_test_loss:
            best_test_loss = avg_loss.data
            myparameter = copy.deepcopy(nn.state_dict())
        mytestloss.append(avg_loss.data)
        
        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.9f} |')
            
    print('Finished.')
    return myloss, myparameter


def GetGradientFor(aapnn, train_loader, m, T, K, lossfunction):
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
    optimizer.zero_grad()
    for n, p in aapnn.named_parameters():
        temp = torch.cat([gradient[n][None, :] for gradient in gradients])
        p.grad = torch.mean(temp, dim=0) / K
    optimizer.step()
    return aapnn


def Test(nn, lossfunction, m, T, test_loader, M, K):
    avg_loss = 0
    avg_acc = 0
    test_time = np.linspace(0,1,K)
    for omega in range(M):
        nn.apply(ps.makemodel)
        for test_t in test_time:
            nn.apply(lambda z: ps.settime(z, test_t))
            for x_test, y_test in test_loader:
                prediction_test = nn(x_test)
                loss_test = lossfunction(prediction_test, y_test, m, T)
                y_hat = torch.argmax(prediction_test, dim=1).data.numpy().squeeze()
                acc_test = sum(y_hat == y_test.numpy()) / y_test.shape[0]

                avg_loss += loss_test.data
                avg_acc += acc_test

    avg_loss /= (M*K)
    avg_acc /= (M*K)
    return avg_loss, avg_acc    


def ParallelTrainingAAPNN(AAPNN, train_loader, test_loader, optimizer, lossfunction, m, T, M, K, M_test, K_test, Epoch):
    test_loss = []
    best_loss = 10000
    parameter = copy.deepcopy(AAPNN.state_dict())
    
    for epoch in tqdm(range(Epoch)):
        Parallel_Models = ps.MakeParallelModels(AAPNN, M)

        gradients = Parallel(n_jobs=M)(delayed(GetGradientFor)(aapnn,
                                                               train_loader,
                                                               m, T, K,
                                                               lossfunction) for aapnn in Parallel_Models)
        UpDateParameter(AAPNN, gradients, K, optimizer)
        loss_temp, acc_temp = Test(AAPNN, lossfunction, m, T, test_loader, M_test, K_test)
        
        if not epoch % 10:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_temp:.5f} | Loss: {loss_temp:.9f} |')
        
        test_loss.append(loss_temp)
        if loss_temp < best_loss:
            parameter = copy.deepcopy(AAPNN.state_dict())
            
            with open('./temp/AAPNN.p', 'wb') as f:
                pickle.dump(AAPNN, f)
        
    return test_loss, parameter
        
        
        
        
        
        
        
        
        
        