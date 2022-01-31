import copy
import pNN_aging_aware_vectorization as pnnv
import torch
from torch.autograd import Variable
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pickle


def train_normal_nn(nn, train_loader, test_loader, optimizer, lossfunction, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytrainloss = []
    mytestloss = []
    averager = [100000]
    best_test_loss = 100000
    
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train)
            loss.backward()
            optimizer.step()
        mytrainloss.append(loss.cpu().data)
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                prediction_test = nn(x_test)
                y_hat = torch.argmax(prediction_test, 1).cpu().data.numpy().squeeze()
                acc_test = sum(y_hat == y_test.cpu().numpy()) / y_test.shape[0]
                loss_test = lossfunction(prediction_test, y_test)

        if loss_test.data < best_test_loss:
            best_test_loss = loss_test.data
            myparameter = copy.deepcopy(nn.state_dict())
        mytestloss.append(loss_test.cpu().data.item())
        
        
        if not epoch % 500:
                print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_test:.5f} | Loss: {loss_test.data:.9f}')
                
        if epoch >= Epoch*0.2:
            if not epoch % 100:
                averager.append(np.mean(mytestloss[-5000::100] * np.linspace(0,1,50)))
            if averager[-2] <= averager[-1]:
                print('Early stop.')
                break
                
    print('Finished.')
    return mytrainloss, mytestloss, myparameter



def train_normal_pnn(nn, train_loader, test_loader, m, T, optimizer, lossfunction, Epoch=500, cache='default'):
    myparameter = copy.deepcopy(nn.state_dict())
    mytrainloss = []
    mytestloss = []
    averager = [100000]
    best_test_loss = 100000
    
    for epoch in tqdm(range(Epoch)):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            
            xv_train = x_train.repeat(1,1,1,1)
            prediction = nn(xv_train)
            
            loss = lossfunction(prediction, y_train, m, T)
            loss.backward()
            optimizer.step()
            
        mytrainloss.append(copy.deepcopy(loss.cpu().data.item()))
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                xv_test = x_test.repeat(1,1,1,1)
                prediction_test = nn(xv_test)

                loss_test = lossfunction(prediction_test, y_test, m, T)

                y_hat = torch.argmax(prediction_test, 3).cpu().flatten().data.numpy().squeeze()
                y_test = y_test.repeat(1,1,1).cpu().flatten().data.numpy().squeeze()
                acc_test = sum(y_hat == y_test) / y_test.shape[0]

        if loss_test.data < best_test_loss:
            best_test_loss = loss_test.data
            myparameter = copy.deepcopy(nn.state_dict())
            
            with open(f'./temp/{cache}_PNN.p', 'wb') as f:
                pickle.dump(nn, f)
                
        mytestloss.append(copy.deepcopy(loss_test.cpu().data.item()))
        
        if not epoch % 500:
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_test:.5f} | Loss: {loss_test.data:.9f} |')
        
        if epoch >= Epoch*0.4:
            if not epoch % 100:
                averager.append(np.mean(mytestloss[-5000::100] * np.linspace(0,1,50)))
            if averager[-2] <= averager[-1]:
                print('Early stop.')
                break
        
    print('Finished.')
    return mytrainloss, mytestloss, myparameter


def train_aged_pnn(nn, train_loader, test_loader, m, T, M, K, M_test, K_test, optimizer, lossfunction, Epoch=500, cache='default'):
    myparameter = copy.deepcopy(nn.state_dict())
    mytestloss = []
    mytrainloss = []
    best_test_loss = 1000
    
    for epoch in tqdm(range(Epoch)):

        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            
            nn.apply(lambda z: pnnv.MakeModel(z, M))
            nn.apply(lambda z: pnnv.SetTime(z, np.random.rand(K).tolist()))
            
            xv_train = x_train.repeat(M,K,1,1)
            prediction = nn(xv_train)

            loss = lossfunction(prediction, y_train, m, T)
            loss.backward()

            optimizer.step()  
            
        mytrainloss.append(loss.cpu().data.item())
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                nn.apply(lambda z: pnnv.MakeModel(z, M=M_test))
                nn.apply(lambda z: pnnv.SetTime(z, np.random.rand(K_test).tolist()))

                xv_test = x_test.repeat(M_test,K_test,1,1)
                prediction_test = nn(xv_test)

                loss_test = lossfunction(prediction_test, y_test, m, T)

                y_hat = torch.argmax(prediction_test, 3).flatten().cpu().data.numpy().squeeze()
                y_test = y_test.repeat(M_test,K_test,1).flatten().cpu().data.numpy().squeeze()
                acc_test = sum(y_hat == y_test) / y_test.shape[0]
        
        if loss_test.data < best_test_loss:
            best_test_loss = loss_test.data
            myparameter = copy.deepcopy(nn.state_dict())
            
            with open(f'./temp/{cache}_AAPNN.p', 'wb') as f:
                pickle.dump(nn, f)
            
        mytestloss.append(loss_test.cpu().data.item())
        
        if not epoch % int(Epoch/20):
            print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_test:.5f} | Loss: {loss_test:.9f} |')
            
    print('Finished.')
    return mytrainloss, mytestloss, myparameter

        