import copy
import pNN_aging_aware as pnn
import torch
from torch.autograd import Variable
import numpy as np

def train_normal_pnn(nn, train_loader, test_loader, m, T, optimizer, Epoch=500):
    myparameter = copy.deepcopy(nn.state_dict())
    mytrainloss = []
    mytestloss = []
    best_test_loss = 100000
    
    for epoch in range(Epoch):
        for x_temp, y_temp in train_loader:
            optimizer.zero_grad()
            prediction = nn(x_temp)
            loss = pnn.LossFunction(prediction, y_temp, m, T)
            loss.backward()
            optimizer.step()
        mytrainloss.append(loss.data)
        
        for x_valid, y_valid in test_loader:
            prediction_valid = nn(x_valid)
            p = torch.argmax(prediction_valid, 1)
            pred_y = p.data.numpy().squeeze()
            acc_valid = sum(pred_y == y_valid.numpy()) / y_valid.shape[0]
            loss_valid = pnn.LossFunction(prediction_valid, y_valid, m, T)

        if loss_valid.data < best_test_loss:
            best_test_loss = loss_valid.data
            myparameter = copy.deepcopy(nn.state_dict())
        mytestloss.append(loss_valid.data)
        print(f'| Epoch: {epoch:-5d} | Accuracy: {acc_valid:.5f} | Loss: {loss_valid.data:.5f} |')
        
    print('Finished.')
    return mytrainloss, mytestloss, myparameter

def train_aged_pnn(nn, train_loader, test_loader, m, T, M, K, optimizer, Epoch=500):
    myparameter = []
    myloss = []
    averager = [1000]
    for epoch in range(Epoch):
        for x_temp, y_temp in train_loader:
            optimizer.zero_grad()

            TimeSet = np.random.rand(K)
            for omega in range(M):
                for l in nn:
                    l.generate_aging_model()
                for k in TimeSet:
                    for i in nn:
                        i.t = k
                    prediction = nn(x_temp)
                    loss = pnn.LossFunction(prediction, y_temp, m, T)
                    loss.backward()

            optimizer.step()

        avg_loss = 0
        avg_acc = 0

        test_time = np.linspace(0,1,10)
        for omega in range(10):
            for l in nn:
                l.generate_aging_model()
            for test_t in test_time:
                for i in nn:
                    i.t = test_t  
                for x_valid, y_valid in test_loader:
                    prediction_valid = nn(x_valid)
                    loss_valid = pnn.LossFunction(prediction_valid, y_valid, m, T)
                    p = torch.argmax(prediction_valid, dim=1)
                    pred_y = p.data.numpy().squeeze()
                    acc_valid = sum(pred_y == y_valid.numpy()) / y_valid.shape[0]

                    avg_loss += loss_valid.data
                    avg_acc += acc_valid

        avg_loss /= 100
        avg_acc /= 100

        myparameter.append(copy.deepcopy(nn.state_dict()))
        myloss.append(avg_loss)
        if epoch > 50:
            averager.append(np.mean(myloss[epoch-50:epoch]))
            if averager[-1] > averager[-2]:
                break
        
        print(f'| Epoch: {epoch:-5d} | Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.5f} |')
            
    print('Finished.')
    return myloss, myparameter
