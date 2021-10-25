import torch

def basic_accuracy(nn, i, t, *args, **kwargs):
    ''' no sensing marigin considere ''' 

    act, idx = torch.max(nn(i), dim=1)

    corrects = (t.view(-1) == idx)

    return corrects.float().sum().item()/t.numel()

def maa(nn, i, t, sensing_margin=.1):
    ''' measure aware accuracy 

        nn : neural network
        i : input x 
        t : target y 
    '''
        
    # get topk and top activations
    act, idx = torch.topk(nn(i), k=2, dim=1)

    # compare activations    top 
    corrects = (act[:, 0] >= sensing_margin) & (act[:, 1] <= 0) & (t.view(-1) == idx[:, 0])
                
    return corrects.float().sum().item()/t.numel()