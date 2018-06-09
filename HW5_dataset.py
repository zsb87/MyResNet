from __future__ import print_function, division
import os
import scipy.io as sio
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import torch.utils.data as utils_data
from PIL import Image
import utils
import warnings
warnings.filterwarnings("ignore")




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



class HW5_dataset(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.mdict = sio.loadmat(self.root)

        self.X = self.mdict['X'].astype('float32').reshape((60000, 784))/255
        self.y = self.mdict['y'].reshape((60000, 1))

        self.XY = np.hstack((self.X,self.y))
        self.XY_train, self.XY_val = tt_split_pseudo_rand(self.XY, 0.9, seed=1)
        self.X = self.XY_train[:,:-1]
        self.Y = self.XY_train[:,-1]
        self.X_val = self.XY_val[:,:-1]
        self.y_val = self.XY_val[:,-1]

        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.Y.squeeze()).long()

        self.X_val = torch.from_numpy(self.X_val)
        self.y_val = torch.from_numpy(self.y_val.squeeze()).long()

        # train_data = utils_data.TensorDataset(X, y)
        # val_data = utils_data.TensorDataset(X_val, y_val)

    def __len__(self):
        if self.train:
        	return len(self.y)
        else:
        	return len(self.y_val)

    def __getitem__(self, index):
        if self.train:
            img, target = self.X[index], self.y[index]
        else:
            img, target = self.X_val[index], self.y_val[index]

        img = Image.fromarray(img.numpy().reshape((28,28)), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target



if __name__== '__main__':

    train_transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            # utils.RandomRotation(),
                            utils.RandomTranslation(),
                            # utils.RandomVerticalFlip(),
                            transforms.ToTensor()
                            # transforms.Normalize((0.1307,), (0.3081,))
                            ]
                            )

    val_transforms = transforms.Compose([
                            transforms.ToTensor()
                            # transforms.Normalize((0.1307,), (0.3081,))
                            ])


    train_data = HW5_dataset('data', train = 0, transform = train_transforms)

    print(train_data[100])

    # for i in range(len(train_data)):
    # 	sample = train_data[i]
    # 	print(i)



    # print(train_data)
    # print(train_data.__dict__)


    # train_loader = utils_data.DataLoader(train_data)

    # print(train_loader)


