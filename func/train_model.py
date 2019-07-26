#!/usr/bin/env python
# -*- coding:utf-8 -*-
#coder:UstarZzz
#date:2019/7/22
"""
to train a model that can recognize
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from utils import dataloader
import cv2
import numpy as np



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      #input_size(3,64,64)
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     #output_size(32,32,32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(                      #input_size(32,32,32)
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output_size(64,16,16)
        )
        self.out = nn.Linear(16*16*64,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        """
        """
        x = x.view(x.size(0),-1)
        output =self.out(x)
        return output

def train():
    # Hyper Parameters
    EPOCH = 10
    BATCH_SIZE = 50
    LR = 0.0001

    # load train_dataset
    train_datasets = dataloader.Data_Creater(path="E:/code/control_system/pic/train")
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)

    # load test_dataset
    test_datasets = dataloader.Data_Creater(path="E:/code/control_system/pic/test",train=False,number=100)
    testloader = Data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=True)

    cnn =CNN()
    optimizer =torch.optim.Adam(cnn.parameters(),lr=LR)
    loss_func =nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(trainloader):
            x = x.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)
            b_x = Variable(x)
            b_y = Variable(y)
            output =cnn(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50 == 0:
                for step, (x, y) in enumerate(testloader):
                    x = x.type(torch.FloatTensor)
                    y = y.type(torch.LongTensor)
                    test_x = Variable(x)
                    test_output = cnn(test_x)
                    test_y = y.data.numpy()
                    pred_y =torch.max(test_output,dim=1)[1].data.numpy().squeeze()
                    accuracy =sum(pred_y==test_y)/test_y.size
                    print('Epoch:',epoch,'|train loss:%.4f'%loss.item(),'|test accuracy:',accuracy)
    torch.save(cnn.state_dict(),'E:/code/control_system/model/cnn.pkl')

def test(all=True,path="E:/code/control_system/pic/test/0 5.jpg",from_camera=True,image=None):
    cnn = CNN()
    cnn.load_state_dict(torch.load('E:/code/control_system/model/cnn.pkl'))
    # Hyper Parameters
    BATCH_SIZE = 50

    if(all==True):
        # load test_dataset
        test_datasets = dataloader.Data_Creater(path="E:/code/control_system/pic/test",train=False,number=1991)
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=True)

        for step, (x, y) in enumerate(testloader):
            x = x.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)
            test_x = Variable(x)
            test_output = cnn(test_x)
            test_y = y.data.numpy()
            pred_y = torch.max(test_output, dim=1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('test accuracy:', accuracy)
    if(all==False):
        if(from_camera == False):
            image = cv2.imread(path)
        if(from_camera == True):
            image =image
        loader = dataloader.Dataloader()
        reshaped_image = loader.reshape(image)
        x = torch.from_numpy(reshaped_image)
        x = x.transpose(0,1)
        x = x.transpose(0,2)
        x = x.numpy()
        data = np.empty((1,3,64,64))
        data[0] = x
        data = torch.from_numpy(data)
        x = data.type(torch.FloatTensor)
        test_x = Variable(x)
        test_output = cnn(test_x)
        mark = int(torch.max(test_output,dim=1)[0].data.numpy().squeeze())
        pred_y = torch.max(test_output, dim=1)[1].data.numpy().squeeze()
        return pred_y,mark


if __name__ == '__main__':
    #train()
    #test()
    for i in range(10):
        prediction,mark = test(all=False, path="E:/code/control_system/pic/test/0 5.jpg", from_camera=False)
        if(mark > 600):
            if(prediction == 0):
                print("刘德华")
            if(prediction == 1):
                print("郑宇星")
        else:
            print("无法识别")