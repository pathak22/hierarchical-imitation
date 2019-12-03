'''
Training file for the inverse model
Contains:
1. Input Data
2. Dataloader
3. Code used for Training
4. Arguments that can be varied (lr, etc)
'''
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
parser.add_argument('--lr', type=float, default=1e-3, help='name of experiment')
parser.add_argument('--batch_size', type= int, default=1, help='Batch size for training')
parser.add_argument('--num_workers', type= float, default=10, help='Numworkers during training')
parser.add_argument('--home_folder', type= str, default='/nfs.yoda/gauravp/pratyusha/third-person-imitation/', help='Where are you training your model')
parser.add_argument('--data_root', type= str, default='/nfs.yoda/gauravp/pratyusha/data/RGB/', help='Where is your data')
parser.add_argument('--data_start', type=int, default=1, help='Input data task numbers from')
parser.add_argument('--data_end', type=int, default=2, help='Input data task numbers to')
parser.add_argument('--remove', type=int, default=30, help='removes the part of the hand approaching')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--img_percent', type=float, default=0.80, help='Percentage of image to jitter')
parser.add_argument('--reshape', type=int, default=64, help='Image to be reshaped to this size')
parser.add_argument('--remove_layers', type=int, default=3, help='How many last layers of the resnet block should be removed')
parser.add_argument('--epochs', type=int, default=500, help='Num of epochs')


args = parser.parse_args()

if not os.path.exists('./ckpts'):
    os.makedirs('./ckpts')

if not os.path.exists(os.path.join('./ckpts', args.exp)):
    os.makedirs(os.path.join('./ckpts', args.exp))

if not os.path.exists('./tbs'):
    os.makedirs('./tbs') 

train_writer = SummaryWriter(os.path.join('./tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./tbs', args.exp, 'val'))

########################################Input Data#####################################################################################################

train_path = args.data_root+'/train/' 
val_path = args.data_root+'/val/' 
train_files = [os.path.join(train_path, subdir) for subdir in os.listdir(train_path)]
val_files = [os.path.join(val_path, subdir) for subdir in os.listdir(val_path)]


train_dataset = Dataloading(train)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

val_dataset = Dataloading(val)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)


#######################################Training code###############################################################################################


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    seq = Controller_NN()
    seq = torch.nn.DataParallel(seq,device_ids=range(torch.cuda.device_count()))
    seq.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(args.epochs):
        print('STEP: ', epoch)
        seq.train()
        avg_train_loss = []
        for i_batch, sample_batched in enumerate(train_dataloader):
            jointangles = sample_batched[1].type(torch.cuda.FloatTensor) # Shape :[Batchsize, timesteps, dimensions]

            optimizer.zero_grad()
            for img_no in range(args.remove,200-args.remove):
                img_1= image_ready(sample_batched[0],img_no).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
                img_2= image_ready(sample_batched[0],img_no+1).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
                out = seq(img_1,img_2)
                jointangle = jointangles[:,img_no+1,:]
                loss = criterion(out,jointangle)
                print('loss:', loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_train_loss.append(loss.cpu().data.numpy())
        train_writer.add_scalar('data/loss', np.mean(avg_train_loss)*57.2958*57.2958, epoch)
                
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': seq.state_dict(),
                'optimizer' : seq.state_dict(),
            },epoch,args.home_folder, args.exp)
     
        seq.eval()
        avg_val_loss = []
        for i_batch, sample_batched in enumerate(val_dataloader):
            jointangles = sample_batched[1].type(torch.cuda.FloatTensor) # Shape :[Batchsize, timesteps, dimensions]
            for img_no in range(args.remove,200-args.remove):
                img_1= image_ready(sample_batched[0],img_no).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
                img_2= image_ready(sample_batched[0],img_no+1).type(torch.cuda.FloatTensor) # Shape :[Batchsize, 3, 224,224]
                out = seq(img_1,img_2)
                jointangle = jointangles[:,img_no+1,:]
                loss = criterion(out,jointangle)
                avg_val_loss.append(loss.cpu().data.numpy())
        val_writer.add_scalar('data/loss', np.mean(avg_val_loss)*57.2958*57.2958, epoch)