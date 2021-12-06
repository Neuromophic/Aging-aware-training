import torch

def basic(nn, x, y, *args, **kwargs):
    ''' no sensing marigin considere ''' 

    act, idx = torch.max(nn(x), dim=1)
    corrects = (y.view(-1) == idx)

    return corrects.float().sum().item()/y.numel()

def maa(nn, x, y, sensing_margin=.01):
    ''' measure aware accuracy 

        nn : neural network
        x : input x 
        y : target label y 
    '''
        
    # get topk and top activations
    act, idx = torch.topk(nn(x), k=2, dim=1)

    # compare activations top 
    corrects = (act[:, 0] >= sensing_margin) & (act[:, 1] <= 0) & (y.view(-1) == idx[:, 0])
                
    return corrects.float().sum().item()/y.numel()