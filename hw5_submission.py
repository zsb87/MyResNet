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

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
import torchvision.transforms as transforms
import scipy.io as sio
from torch.autograd import Variable
from datetime import datetime
import scipy.misc
import utils
import random
import model
from HW5_dataset import HW5_dataset
import argparse



def create_folder(f, deleteExisting=False):
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def lprint(logfile, *argv):
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


def train(model, train_loader, criterion, optimizer):
    # set to training mode
    model.train()
    running_loss = 0
    running_correct = 0

    for i, data in enumerate(train_loader):
        X, y = data
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        # zero gradient    
        optimizer.zero_grad()

        # forward pass and compute loss
        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        running_correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    train_acc = running_correct.numpy()/len(train_loader.dataset)

    return running_loss/len(train_loader), train_acc


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0
    running_correct = 0
    for i, (X,y) in enumerate(val_loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)
        output = model(X)
        loss = criterion(output, y)

        running_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        running_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
    
    val_acc = running_correct.numpy()/len(val_loader.dataset)

    return running_loss/len(val_loader), val_acc




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet18', help="model")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--nepochs", type=int, default=400, help="max epochs")
    parser.add_argument("--nocuda", action='store_true', help="no cuda used")
    parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    cuda = not args.nocuda and torch.cuda.is_available()

    # set seed
    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            # utils.RandomRotation(),
                                            utils.RandomTranslation(),
                                            # utils.RandomVerticalFlip(),
                                            transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = HW5_dataset('data', train = 1, transform = train_transforms)
    val_data = HW5_dataset('data', train = 0, transform = val_transforms)



    train_loader = utils_data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.nworkers)
    val_loader = utils_data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.nworkers)    
     



    net = model.__dict__[args.model]()
    print(net)

    
    # Change optimizer for finetuning    
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        net, criterion = net.cuda(), criterion.cuda()


    for epoch in range(args.nepochs):
        train_loss, train_acc = train(net, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(net, val_loader, criterion)

        print('Epoch: {}  Train Loss: {:.3f}  Training Accuracy: {:.3f}  Val loss: {:.3f}, Val acc: {:.3f}'.format(epoch, 
                train_loss, train_acc, val_loss, val_acc))

        if val_acc > 0.8:
            torch.save({'arch': args.model,'state_dict': net.state_dict()},
                        'trained_model_{}_epoch{}_acc{}.pt'.format(args.model, epoch, val_acc))

