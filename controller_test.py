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
from inverse_models import Controller_NN

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='name of experiment')
parser.add_argument('--batch_size', type= int, default=1, help='Batch size for training')
parser.add_argument('--num_workers', type= float, default=10, help='Numworkers during training')
parser.add_argument('--home_folder', type= str, default='/location/of/home_folder/', help='Where are you training your model')
parser.add_argument('--data_root', type= str, default='/location/of/data_root/', help='Where is your data')
parser.add_argument('--data_start', type=int, default=1, help='Input data task numbers from')
parser.add_argument('--data_end', type=int, default=2, help='Input data task numbers to')
parser.add_argument('--remove', type=int, default=30, help='removes the part of the hand approaching')
parser.add_argument('--img_percent', type=float, default=0.80, help='Percentage of image to jitter')
parser.add_argument('--reshape', type=int, default=64, help='Image to be reshaped to this size')
parser.add_argument('--remove_layers', type=int, default=3, help='How many last layers of the resnet block should be removed')
parser.add_argument('--chkpt', type= str, default='/location/of/chkpt/',help='Location of the checkpoint to load')

args = parser.parse_args()

########################################Input Dat a#####################################################################################################

test_path = args.data_root+'/test/' 

test_files = [os.path.join(test_path, subdir) for subdir in os.listdir(test_path)]


test_dataset = Dataloading(test)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
#######################################Test code###############################################################################################


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    seq = Controller_NN()
    seq = torch.nn.DataParallel(seq,device_ids=range(torch.cuda.device_count()))
    seq.cuda()
    checkpoint = torch.load(args.chkpt)
    seq.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss()
    
    seq.eval()
    avg_test_loss = []
    for i_batch, sample_batched in enumerate(test_dataloader):
        jointangles = sample_batched[1].type(torch.cuda.FloatTensor) # Shape :[Batchsize, timesteps, dimensions]
        for img_no in range(args.remove,200-args.remove):
            img_1= image_ready(sample_batched[0],img_no).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
            img_2= image_ready(sample_batched[0],img_no+1).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
            out = seq(img_1,img_2)
            jointangle = jointangles[:,img_no+1,:]
            loss = criterion(out,jointangle)
            avg_test_loss.append(loss.cpu().data.numpy())
print(avg_test_loss)
