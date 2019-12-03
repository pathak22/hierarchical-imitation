#file containing the inverse models tried out
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import pickle
from IPython import embed 
from tensorboardX import SummaryWriter
import argparse
import random
import os.path
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image
import time
import cv2
import shutil
from collections import OrderedDict
from ast import literal_eval


class Controller_NN(nn.Module):
    def __init__(self):
        super(Controller_NN, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.modified_pretrained = nn.Sequential(*list(resnet18.children())[:-1])
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512*2,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,128)
        self.linear4 = nn.Linear(128,64)
        self.linear5 = nn.Linear(64, 32)
        self.linear = nn.Linear(32, 14)

    
    def forward(self, input1,input2):
        i_1 = self.modified_pretrained(input1)
        i_2 = self.modified_pretrained(input2)
        i_combined = torch.cat((i_1,i_2),1)
        i1 = self.relu(self.linear1(i_combined.squeeze(2).squeeze(2)))
        i2 = self.relu(self.linear2(i1))
        i3 = self.relu(self.linear3(i2))
        i4 = self.relu(self.linear4(i3))
        h_t = self.linear5(i4)
        output = self.linear(h_t)
        return output


class Controller_LSTM(nn.Module):
    def __init__(self):
        super(Controller_LSTM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.modified_pretrained = nn.Sequential(*list(resnet18.children())[:-3])
        self.avgpool = nn.AvgPool2d(kernel_size=int(7*(2**(3-2))*64/224), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512*2/(2**(3-2)),256)
        #self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,128)
        self.linear4 = nn.Linear(128,64)
        self.lstm = nn.LSTMCell(64, 32)
        self.linear = nn.Linear(32, 14)
    
    def forward(self, input1,input2,h,t):
        h_t = h
        c_t = t
        i_1 = self.avgpool(self.modified_pretrained(input1))
        i_2 = self.avgpool(self.modified_pretrained(input2))
        i_combined = torch.cat((i_1,i_2),1)
        i1 = self.relu(self.linear1(i_combined.squeeze(2).squeeze(2)))
        #i2 = self.relu(self.linear2(i1))
        i3 = self.relu(self.linear3(i1))
        i4 = self.relu(self.linear4(i3))
        h_t, c_t = self.lstm(i4, (h_t, c_t))
        output = self.linear(h_t)
        return output, h_t, c_t

