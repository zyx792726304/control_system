# control_system
recognize the face of human,and use your arm to control
This is a code for exercise,and I just write it to practice my abilities for coding.
If you want to make it work,you can do the following steps:
1.Connect you camera and run func/recog_save.py to save your pictures in your specific folder.You can use:
      frame(path="your folder",save=True,number=2000)
2.Create a folder(name"pic" for example),and create "train"folder and "test"folder in folder"pic"
3.Run utils/convert.py to rename your pictures,and then move it to "test"folder or "train"folder
4.Run func/train_model.py to train your model,you can use the following code in main:
      train() # to train your model
      test() # to test the accuracy
      for i in range(10):
        prediction,mark = test(all=False, path="E:/code/control_system/pic/test/0 5.jpg", from_camera=False)
        if(mark > 600):
            if(prediction == 0):
                print("0")
            if(prediction == 1):
                print("1")
        else:
            print("Can not recognize")
5.Run func/recog_save.py,and use this code in main:
      frame()
  and it will start to recognize your face,if you have trained your pictures,it will put text"your name" to the window,and if you haven't
trained the pictures before,it will take no action.


That is the whole steps for my code.I will keep upgrading my code for further development.
In the next days,I will do:
1.Add TensorFlow/Keras code
2.Try to fulfill the function of posture-estimation
