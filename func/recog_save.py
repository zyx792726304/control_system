#!/usr/bin/env python
# -*- coding:utf-8 -*-
#coder:UstarZzz
#date:2019/7/22
"""
    to show the picture in the video and frame out the face
    if you want to save picture to train your model,you can use frame(path=your_path,save=True,number=1000)
"""

def frame(path='E:/code/control_system/pic/test_zyx',save=False,number=1000,from_camera=True,video_path="E:/code/control_system/video/video_train.mp4"):
    from utils import recognizer

    recognizer = recognizer.Recognizer(0, "F:/study/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml")
    cap = recognizer.get_cap(from_camera=from_camera,path=video_path)
    num = 0
    while cap.isOpened():
        state,frame = recognizer.get_video(cap)
        if not state:
            break
        recognizer.rect_face(frame,num,save=save,path=path)
        num = num + 1
        if(num>number):
            save = False
        keycode = recognizer.show_pic(frame)
        if keycode&0xff == ord('q'):
            break
    recognizer.release(cap)

if __name__ == '__main__':
    #if you want to save picture to a specific place
    #please use frame(your_path,save=True,number=your_number)
    #if you want to save picture in the outside video,please use the following code:
    #frame(path='E:/code/control_system/pic/test_oth',save=True,number=1000,from_camera=False,video_path="")
    #or if you just want to see the outside video,please use:
    #frame(from_camera=False,video_path="")


    # frame(path='E:/code/control_system/pic/train_oth',save=True,number=2000,
    #       from_camera=False,video_path="E:/code/control_system/video/video_train.mp4")
    #frame(path='E:/code/control_system/pic/zyx_pic',save=True,number=2500)

    #if you just want to use camera to frame out yourself,please use frame()
    frame()
