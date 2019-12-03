# Dataloader
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
import utils
import dataloader

class Dataloading(Dataset):
    def __init__(self,name):
        self.name = name

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        robot_images = self.name[idx]+'/robot_img/'
        with open(self.name[idx]+'/jointangles') as f:
            joint_angles = f.read().splitlines()
        return (robot_images,preprocess_jointangles(joint_angles))
