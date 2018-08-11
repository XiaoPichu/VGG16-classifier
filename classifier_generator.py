#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August  1 10:30:04 2018

@author: crunch
python3 类的使用 写法跟python2有点不同

数据集的文件结构:
data/
      train/
            dog/
            cat/
      valid/
            dog/
            cat/
      test/
            文件名.jpg 文件名.jpg ...
"""

from keras.applications.imagenet_utils import preprocess_input  #### 会把RGB 图像的像素点的取值范围从0-255归一化到0-1 便于网络学习
import numpy as np
import os
from random import shuffle
from scipy.misc import imread, imresize  #### 这两个要求提前安装PIL或者Pillow

class MyGenerator(object):
      
    #### 定义一些类的变量 通过self.变量名调用
    def __init__(self, path_traindatas, path_validdatas,patch_size,batch_size,classnames,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.train_datas = []                                          #### train_datas 中存放了图片路径
        fs = os.listdir(path_traindatas)
        for f in fs:
            for p in os.listdir(os.path.join(path_traindatas,f)):
                tmp_path = os.path.join(path_traindatas,f,p)
                self.train_datas.append(tmp_path)            
        self.train_keys = list(range(0,len(self.train_datas))) 
        self.steps_per_epoch = int(len(self.train_keys) / batch_size)
        self.valid_datas = []
        fs = os.listdir(path_validdatas)
        for f in fs:
            for p in os.listdir(os.path.join(path_validdatas,f)):
                tmp_path = os.path.join(path_validdatas,f,p)
                self.valid_datas.append(tmp_path)
        self.valid_keys = list(range(0,len(self.valid_datas)))
        self.train_batches = len(self.train_keys)//self.batch_size
        self.valid_batches = len(self.valid_keys)//self.batch_size
        #### 控制图像增强的一些参数
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var
        self.contrast_var = contrast_var
        self.lighting_std = lighting_std

      
    #### 定义一些类的函数，self只在函数定义的时候传进去，证明是类的函数，可以被调用 但是 在调用的时候不用传递
    #### 数据增强部分 只对训练集做数据增强，验证集不需要
    def grayscale(self, rgb):                  #### 得到灰度图
        return rgb.dot([0.299, 0.587, 0.114])  #### rgb转灰度图
    def saturation(self, rgb):                 #### 饱和度变换
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)
    def brightness(self, rgb):                 #### 亮度变换
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)
    def contrast(self, rgb):                   #### 对比度变换
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)
    def lighting(self, img):                   #### 明暗度变换
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    def horizontal_flip(self, img):           #### 图像水平翻转
        img = img[:, ::-1]
        return img
    def vertical_flip(self, img):             #### 图像上下翻转
        img = img[::-1]
        return img
    #### 0.25的概率返回原图 0.125的概率返回saturation，brightness，contrast，lighting，horizontal_flip，vertical_flip
    def image_augument(self,img):
        mode_prob = np.random.random()         #### 生成一个0-1内的随机数
        if mode_prob<0.25:
            return img
        elif mode_prob<0.375:
            return self.saturation(img)
        elif mode_prob<0.5:
            return self.brightness(img)
        elif mode_prob<0.625:
            return self.contrast(img)
        elif mode_prob<0.75:
            return self.lighting(img)
        elif mode_prob<0.875:
            return self.horizontal_flip(img)
        else:
            return self.vertical_flip(img)
    #### onehot编码方式：将 类别0 由 [0] 变成 [1,0],将 类别1 由 [1] 变成 [0,1]
    def one_hot(self,y_filename):
        class_array = np.eye(self.num_classes)               #### 生成一个只有主对角线元素为1其余为0的
                                                             #### self.num_classes*self.num_classes 大小的浮点数矩阵
        #### python中有两个很重要的用法 lamda 和正则表达式re
        #### lamda 将函数体转换为一行函数  func=lambda x:x+1 等同于 def func(x):  return(x+1)
        #### re 匹配时候用到
        array_index = [i for i in range(self.num_classes) if self.classnames[i] in y_filename]#### 如果图片路径中会有类别的文件夹的字符 跟 classnames 中有重合的即为这张图片label
        #array_index = filter(lambda x: classnames[x] in y_filename, range(self.num_classes))
        return class_array[array_index][0]  # Returns FLOATS
    #### 产生数据
    def generate(self, trainflag):
        if trainflag == True:
            datas = self.train_datas
            keys = self.train_keys
            num_batch = self.train_batches
        else:
            datas = self.valid_datas
            keys = self.valid_keys
            num_batch = self.valid_batches
        while True:
            shuffle(keys)
            for i in range(num_batch):      
                inputs = []
                targets = []
                for j in range(self.batch_size):
                    tmp_imgpath = datas[keys[j + i * self.batch_size]]
                    tmp_img = imread(tmp_imgpath).astype('float32')
                    img = imresize(tmp_img, self.patch_size).astype('float32')
                    if trainflag:
                        img = self.image_augument(img)
                    y = self.one_hot(tmp_imgpath)
                    inputs.append(img)
                    print(j + i * self.batch_size)
                    targets.append(y)
                final_inputs = np.array(inputs)
                targets = np.array(targets)
                yield (preprocess_input(final_inputs), targets)
    '''
    类中无需定义固定会有的几个函数
    def __call__():  #通过 MyGenerator() 默认调用这个函数
    '''
