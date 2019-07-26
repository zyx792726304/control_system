#!/usr/bin/env python
# -*- coding:utf-8 -*-
#coder:UstarZzz
#date:2019/7/22

"""
what this code do is to reshape the picture to (64,64,3) so as to satisfy the input,and it returns
ndarray(num,64,64,3)
"""

import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils import data
import torch
from random import choice
"""
This class is to change the shape of picture to (64,64,3),and it returns the number of images
"""
class Dataloader():
    def __init__(self,path="E:/code/control_system/pic/train"):
        self.path = path


    """
    to calculate the size which need to be added to the picture
    :parameter:h,w
    :return:(top,bottom,left,right)
    """
    def calc_size(self,h,w):
        top,bottom,left,right = (0,0,0,0)
        l = max(h,w)
        if h < l:
            dh = l - h
            top = dh // 2
            bottom = dh-top
        if w < l:
            dw = l - w
            left = dw // 2
            right = dw - left
        return top,bottom,left,right


    """
    to reshape the picture
    :parameter:image
    :return:reshaped picture(64,64)
    """
    def reshape(self,image):
        h, w, _ = image.shape
        top,bottom,left,right = self.calc_size(h,w)

        #to add balck block to the picture
        #cv2.BORDER_CONSTANT means the color of the block,it depends on the value
        constant = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
        #reshape the picture to (64,64)
        return cv2.resize(constant,(64,64))


    """
    to read the picture in the given path
    :parameter:none
    :returns:images,labels
    """
    def load_pic(self):
        images = []
        for dir_item in os.listdir(self.path):
            abspath = os.path.abspath(os.path.join(self.path,dir_item))
            if dir_item.endswith('.jpg'):
                image = cv2.imread(abspath)
                image = self.reshape(image)
                if(int(dir_item.split(' ')[0])==1):
                    images.append((image,1))
                if(int(dir_item.split(' ')[0])==0):
                    images.append((image,0))
        return images


    """
    to load dataset
    :parameter:none
    :return:images(num,3,64,64),number
    """
    def load_dataset(self):
        images = self.load_pic()
        number = len(images)
        return images,number


"""
This class is to make your own dataset
train==True:you are loading train_data
train==False:you are loading test_data
number:the number of the picture in the test_dataset
"""

class Data_Creater(torch.utils.data.Dataset):
    def __init__(self,path="E:/code/control_system/pic/train",train=True,number=100):
        super(Data_Creater, self).__init__()
        self.dataloader =Dataloader(path)
        self.pic,self.number = self.dataloader.load_dataset()
        if(train==True):
            self.images = self.pic
        if(train==False):
            self.images = []
            for i in range(number):
                self.images.append(choice(self.pic))
        self.transforms = transforms.Compose([
                    transforms.ToTensor()
                ])
    def __getitem__(self, index):
        img, label = self.images[index]
        if self.transforms:
            data = self.transforms(img)
        else:
            t_img = np.asarray(img)
            data = torch.from_numpy(t_img)
        return data,label
    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    train_datasets = Data_Creater(path="E:/code/control_system/pic/train")
    print(len(train_datasets))