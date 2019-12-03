# File containing the helper functions

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

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.80, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def image_ready(array_of_path,image_no):
    final_array = data_transforms(default_loader(array_of_path[0]+str(image_no)+'.jpg')).unsqueeze(0)
    if len(array_of_path)>1:
        for i in range(1,len(array_of_path)):
            final_array = torch.cat((final_array,data_transforms(default_loader(array_of_path[i]+str(image_no)+'.jpg')).unsqueeze(0)),0)
    return (final_array)

#Converting array of dictionaries into array of arrays
def preprocess_jointangles(sequence):
    seq_array = []
    for index in range(0,len(sequence)):
        sub_seq = literal_eval(sequence[index])
        temp = np.zeros(14)
        k = ["left_w0", "left_w1", "left_w2", "right_s0", "right_s1", "right_w0", "right_w1", "right_w2", "left_e0", "left_e1", "left_s0", "left_s1", "right_e0", "right_e1"]
        if len(sub_seq)>14:
            for i in range(0,len(k)):
                temp[i] = sub_seq[k[i]]
        seq_array.append(np.copy(temp))
    return np.asarray(seq_array)

def save_checkpoint(state,epoch,home_folder, exp,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, home_folder+'ckpts/'+exp+ '/'+ str(epoch)+'.pth.tar')

def postprocess_jointangles(sequence):
    seq_array = {}
    for index in range(0,len(sequence[0])):
        k = ["left_w0", "left_w1", "left_w2", "right_s0", "right_s1", "right_w0", "right_w1", "right_w2", "left_e0", "left_e1", "left_s0", "left_s1", "right_e0", "right_e1"]
        seq_array[k[index]] = sequence[0][index]
    return seq_array

def write_dic(dic,path):
    f = open(path,"w")
    f.write( str(dic) )
    f.close()

