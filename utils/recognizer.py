#!/usr/bin/env python
# -*- coding:utf-8 -*-
#coder:UstarZzz
#date:2019/7/19

import cv2
from func import train_model

class Recognizer():
    def __init__(self,cam_id,address):
        self.id = cam_id
        self.classfier = cv2.CascadeClassifier(address)
        self.color = (0,255,0)


    """
    to get capture object
    :parameter:none
    :return:cap
    if you want to use camera to capture pictures,you just need to use
    get_cap()
    but if you want to capture pictures in your specific video files,you should use
    get_cap(from_camera=False,path='your own path')
    """
    def get_cap(self,from_camera=True,path='video.mp4'):
        cv2.namedWindow('recognition area')
        if(from_camera == True):
            cap = cv2.VideoCapture(self.id)
        if(from_camera == False):
            cap = cv2.VideoCapture(path)
        return cap


    """
    to get state,frame
    :parameter:cap
    :returns:state,frame
    """
    def get_video(self,cap):
        state,frame = cap.read()
        return state,frame


    """
    to frame the portrait in the picture
    :parameter:frame
    :return:none
    """
    def rect_face(self,frame,num,save=False,path='E:/code/control_system/pic/zyx_test'):
        #change RGB to GREY
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        facerects = self.classfier.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(facerects)>0:
            for facerect in facerects:
                x,y,w,h = facerect
                if(save == True):
                    img_name = '%s/%d.jpg' % (path, num)
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.imwrite(img_name, image)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                if(save == False):
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    prediction,mark = train_model.test(all=False,from_camera=True,image=image)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),self.color,2)
                    if(mark > 600):
                        if(prediction == 0):
                            cv2.putText(frame, 'LiuDehua', (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                        if(prediction == 1):
                            cv2.putText(frame, 'ZhengYuxing', (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                    else:
                        pass


    """
    to show the picture
    :parameter:frame
    :return:keycode
    """
    def show_pic(self,frame):
        cv2.imshow('recognition area',frame)
        keycode = cv2.waitKey(10)
        return keycode


    """
    to release the window
    :parameter:cap
    :return:none
    """
    def release(self,cap):
        cap.release()
        cv2.destroyAllWindows()