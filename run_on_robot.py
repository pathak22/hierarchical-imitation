#!/usr/bin/env python

import sys
import cv2
import rospy
import re
import time
import ast  
import json
import os
import rosbag
import torch

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from baxter_core_msgs.msg import EndEffectorState
from baxter_core_msgs.msg import DigitalIOState
from cv_bridge import CvBridge, CvBridgeError
from torchvision import datasets, models, transforms
from PIL import Image as Im2


from pix2pix import TestOptions
from pix2pix import util
from pix2pix import AlignedDataset
from pix2pix import create_model
from pix2pix import CreateDataLoader
from model import inverse_model

from IPython import embed

import baxter_interface

from baxter_interface import(
        DigitalIO,
        Gripper,
        CHECK_VERSION,
        Navigator
        )

import numpy as np
import glob


##----------------------------------------------Helper Functions ----------------------------------------------##
class Playback:
    def __init__(self):
        self.right_arm = baxter_interface.Limb("right")
        self.left_arm = baxter_interface.Limb("left")
        self.left_keys = [ 'left_e0','left_e1','left_s0','left_s1','left_w0','left_w1','left_w2' ]
        self.right_keys =['right_e0','right_e1','right_s0','right_s1','right_w0','right_w1','right_w2' ]
        self.bridge = CvBridge()
        self.kinectRGB = rospy.Subscriber("/camera/rgb/image_color_throttle",Image, self.image_callback)


    def image_callback(self,msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            cv2.imwrite('/location/to/save/presentimg/present.jpg', cv2.resize(cv2_img[240:480,170:470,:],(300,300)))


    def perform_action(self, dictAngles):
        l = dict((k, dictAngles[k]) for k in self.left_keys if k in dictAngles)
        r = dict((k, dictAngles[k]) for k in self.right_keys if k in dictAngles)
        self.left_arm.set_joint_positions(l)
        self.right_arm.set_joint_positions(r)
        rospy.sleep(0.3)
        


def Pix2Pix(opt):
    model = create_model(opt)
    model.setup(opt)
    return model


def obtain_robot_img(model_1, dataset,time_step):
    # get data somehow
    data = dataset[time_step]
    data['A'] = data['A'].unsqueeze(0)
    data['B'] = data['B'].unsqueeze(0)
    model_1.set_input(data)
    model_1.test()
    print('Reading for model1')
    print(data['A_paths'])
    return model_1.fake_B

reshape = 64
img_percent = 0.8
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(reshape,scale=(img_percent, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def postprocess_jointangles(sequence):
    seq_array = {}
    for index in range(0,len(sequence[0])):
        k = ["left_w0", "left_w1", "left_w2", "right_s0", "right_s1", "right_w0", "right_w1", "right_w2", "left_e0", "left_e1", "left_s0", "left_s1", "right_e0", "right_e1"]
        seq_array[k[index]] = sequence[0][index]
    return seq_array

#------------------------------------------------- End of helper functions ----------------------------------------------##

def run_model_pipelined(cmd_opt):
    '''
    Takes the output of Pix2Pix feeds it to inverse_model
    '''
    model_1 = Pix2Pix(cmd_opt) #the cmd_opt includes which checkpoint needs to be loaded for the goal generator
    model_2 = inverse_model(3, reshape)

    # initialize model_2 with pretrained weights
    model_2 = torch.nn.DataParallel(model_2,device_ids=range(torch.cuda.device_count()))
    model_2.float()
    checkpoint = torch.load('/location/of/the/best/checkpoint/chkpt_number.pth.tar', map_location='cpu')
    model_2.load_state_dict(checkpoint['state_dict'])

    model_2.eval()
    playback = Playback()

    NUM_STEPS_ST = 0
    NUM_STEPS_EN = 100

    dataset = AlignedDataset()   
    dataset.initialize(cmd_opt)


    for i in range(NUM_STEPS_ST, NUM_STEPS_EN):
        #Update the present image in the test dataset
        robot_present = cv2.imread('/home/dhiraj/action_types/present.jpg')
        im = cv2.imread('/location/of/the/dataset/test/filename_{}.jpg'.format(i)) 
        im[:,300:600,:] = robot_present
        cv2.imwrite('/location/to/save/test/filename_{}.jpg'.format(i), im)
        time.sleep(2)

        # Goal Generator: Hallucinate next image 
        robot_next = obtain_robot_img(model_1, dataset, i)
        robot_next = cv2.resize(util.tensor2im(robot_next),(300,300))[:,:,::-1]

        #Inverse-model: Use the present image and hallucinated image to decide action
        robot_next = data_transforms(Im2.fromarray(np.uint8(robot_next))).unsqueeze(0)
        robot_present = data_transforms(Im2.fromarray(robot_present)).unsqueeze(0)
        joint_angles = model_2(robot_present,robot_next)

        # Command the action to the robot
        joint_angles_processed = postprocess_jointangles(joint_angles.cpu().data.numpy())
        playback.perform_action(joint_angles_processed)
        time.sleep(2) 


##------------------------------------------------------MAIN---------------------------------------------------------##
def main(args):

    rospy.init_node("capture_images", anonymous=True)
    opt = TestOptions().parse()
    run_model_pipelined(opt)

if __name__ == '__main__':
    main(sys.argv)
