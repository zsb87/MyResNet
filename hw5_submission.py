"""
Homework 5
IEMS 455: Machine Learning

Shibo Zhang
May 23, 2018

Submission Instructions:

Please submit this file with your stored network weights under trained_model.pt 
file and a .pdf file explaining your network and results. 
    


# X size: torch.Size([60000, 28, 28])
# y set: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
import scipy.io as sio
from datetime import datetime
import scipy.misc
import random
from hack import test_model_hack
from net import Net
import os

def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.

    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')

    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 
        
    Return
    ------
        none

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
        argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    with open(logfile, 'a') as out:
        out.write(str(datetime.now()) +': ' + argAll + '\n')


def tt_split_pseudo_rand(XY, train_ratio, seed):
    # eg: train_ratio = 0.7

    numL = list(range(10))
    random.seed(seed)
    random.shuffle(numL)

    length = len(XY)
    test_enum = numL[0:10-int(10*train_ratio)]
    test_ind = []

    for i in test_enum:
        test_ind = test_ind + list(range(i, length, 10))

    train_ind = [x for x in list(range(length)) if x not in test_ind]

    return XY[train_ind], XY[test_ind]


# set seed
seed = 1226
np.random.seed(seed)
torch.manual_seed(seed)
#%% Parameters

epochs = 10
lr = 0.001
momentum = 0.9
batch_size = 128

logfile = 'log_val.txt'
lprint(logfile, 'seed: ', seed)
lprint(logfile, 'epochs: ', epochs)
lprint(logfile, 'lr: ', lr)
lprint(logfile, 'momentum: ', momentum)
lprint(logfile, 'batch_size: ', batch_size)


#%% Load Data
#
#   It may be beneficial to consider how to transform your data to obtain 
#   better performance.

mdict = sio.loadmat('data')

X = mdict['X'].astype('float32').reshape((60000, 784))/255
y = mdict['y'].reshape((60000, 1))

XY = np.hstack((X,y))
XY_train, XY_val = tt_split_pseudo_rand(XY, 0.9, seed)
X_train = XY_train[:,:-1]
Y_train = XY_train[:,-1]
X_val = XY_val[:,:-1]
y_val = XY_val[:,-1]

X = torch.from_numpy(X_train)
y = torch.from_numpy(Y_train.squeeze()).long()

X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val.squeeze()).long()


train_data = utils_data.TensorDataset(X, y)

# #%% Define Neural Network Architecture
# #
# #   Modify your neural network here!

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 1000)
#         self.fc2 = nn.Linear(1000, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
    
#%% Define Training Function
#
#   Modify your optimization algorithm and learning rate schedule here!

def train_model(model, nb_epochs=10, batch_size=100, lr=0.001, momentum=0.9):
    """
    Trains the model using SGD.
    
    Inputs:
        model: Neural network model
        nb_epochs: number of epochs (int)
        batch_size: batch size (int)
        lr: learning rate/steplength (float)
    
    """
    
    # initialize train loader, optimizer, and loss
    train_loader = utils_data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
        
    for e in range(nb_epochs):

        # Training loop!
        for i, data in enumerate(train_loader):
            
            # get inputs
            inputs, labels = data
            
            # set to training mode
            model.train()
            
             # zero gradient
            optimizer.zero_grad()
            
            # forward pass and compute loss
            ops = model(inputs)
            loss_fn = loss(ops, labels)
            
            # compute gradient and take step in optimizer
            loss_fn.backward() 
            optimizer.step()
        
        model.eval() # set to evaluation mode
        
        # evaluate training loss and training accuracy
        ops = model(X)
        _, predicted = torch.max(ops.data, 1)
        train_loss = loss(ops, y).item()
        train_acc = torch.mean((predicted == y).float()).item()*100
                    
        lprint(logfile, 'Epoch: ', e+1, 'Train Loss: ', train_loss, '  Training Accuracy: ', train_acc)
        
"""
                #%% Train Neural Network
"""

# # Create neural network
# model = Net()

# # Train model
# train_model(model, nb_epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum)

# # Save model weights
# create_folder('m1')
# torch.save(model.state_dict(), './m1/trained_model.pt')

#%% Define Testing Function

def test_model(X, y):
    """
    Tests the model using stored network weights. 
    Please ensure that this code will allow me to test your model on testing data.
    An example code is given below.
    
    Inputs:
        X: feature data (FloatTensor)
        y: labels (LongTensor)
    
    """
    
    # constructs model
    model = Net()
    loss = torch.nn.CrossEntropyLoss()
    
    # loads weights
    # model.load_state_dict(torch.load('./m1/trained_model.pt'))
    mfile = 'trained_model_resnet18_19_epoch0_acc0.5356709788179701.pt'
    mdir = '../pallas'

    model.load_state_dict(torch.load(os.path.join(mdir,mfile)))
    # compute loss and accuracy
    ops = model(X)
    _, predicted = torch.max(ops.data, 1)
    test_loss = loss(ops, y).item()
    test_acc = torch.mean((predicted == y).float()).item()*100
    
    lprint(logfile, 'Test Loss: ', test_loss)
    lprint(logfile, 'Test Accuracy: ', test_acc)


test_model(X_val, y_val)



"""
HACK
"""
test_model_hack()
