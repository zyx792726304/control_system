#!/usr/bin/env python
# -*- coding:utf-8 -*-
#coder:UstarZzz
#date:2019/7/25
"""
This code is to change the name of the picture in your folder to your specific name
"""
import os
import os
class BatchRename():
  def __init__(self):
    self.path = 'E:/code/control_system/pic/zyx_train'
  def rename(self):
    filelist = os.listdir(self.path)
    total_num = len(filelist)
    i = 0
    for item in filelist:
      if item.endswith('.jpg'):
        src = os.path.join(os.path.abspath(self.path), item)
        dst = os.path.join(os.path.abspath(self.path), str(1) + ' ' + str(i) + '.jpg')
        os.rename(src, dst)
        print('converting %s to %s ...' % (src, dst))
        i = i + 1
    print('total %d to rename & converted %d jpgs' % (total_num, i))
if __name__ == '__main__':
  demo = BatchRename()
  demo.rename()