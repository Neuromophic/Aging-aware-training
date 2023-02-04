import torch
import numpy as np
import random
import os

def save_checkpoint(epoch, model, optimizer, loss, setup, path):
    filename = f'{path}/{setup}.ckp'
    if not os.path.exists(path):
        os.makedirs(path)

    random_state = {'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.random.get_rng_state()}
    checkpoint = {'epoch': epoch, 'model': model, 'optimizer': optimizer, 'loss': loss, 'random_state': random_state}
    torch.save(checkpoint, filename)
    return None

def load_checkpoint(setup, path):
    if os.path.isfile(f'{path}/{setup}.ckp'):
        checkpoint = torch.load(f'{path}/{setup}.ckp')
        random.setstate(checkpoint['random_state']['random'])
        np.random.set_state(checkpoint['random_state']['numpy'])
        torch.random.set_rng_state(checkpoint['random_state']['torch'])
        epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        loss = checkpoint['loss']
        return epoch+1, model, optimizer, loss
    else:
        return None
    
def record_checkpoint(epoch, model, train_loss, valid_loss, setup, path):
    filename = f'{path}/{setup}_epoch_{epoch}.ckp'
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint = {'epoch': epoch, 'model': model, 'train_loss': train_loss, 'valid_loss': valid_loss}
    torch.save(checkpoint, filename)
    return None

def load_recorded_checkpoint(epoch, setup, path):
    if os.path.isfile(f'{path}/{setup}_epoch_{epoch}.ckp'):
        checkpoint = torch.load(f'{path}/{setup}_epoch_{epoch}.ckp')
        epoch = checkpoint['epoch']
        model = checkpoint['model']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        return epoch, model, train_loss, valid_loss
    else:
        return None