#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August  1 10:30:04 2018

@author: crunch
一个二分类的网络
"""

from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'               #### 多块显卡的话,指定使用第几块显卡
import argparse                                       #### 命令行传参数 看下面写的 args_parse 函数
import sys                                           #### sys.argv跟argparse可以实现相同的功能
import matplotlib.pyplot as plt                      #### 跟MATLAB里面的plt使用方法一样 画图用
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPool2D, Flatten, Dense,Activation

from classifier_generator import MyGenerator        #### 自己写的一个产生数据的类，封装成类会方便统一调用 可以看完整个流程再回头看他

#### 定义一些全局变量
classnames = ['dog','cat']
#NUM_CLASSES = 2                                     #### 二分类
NUM_CLASSES = len(classnames)                       #### len: 得到列表的长度
EPOCHES = 10                                        #### 全部数据训练完一遍为一代
BATCH_SIZE = 32                                     #### 一批送进多少个数据
PATCH_SIZE = (224,224)
LR = 3e-4                                           #### learning rate，希望它在模型前期大，以快速找到收敛域，后期小得以找到最优解

#### 定义一些函数
###################################################  network structure
def Network():               #### VGG16 经典的图像分类网络
    inputs = Input(shape=PATCH_SIZE)                #### 大小是224*224

    #### 一般这个叫做一个block 因为参数都是32
    conv1_1 = Conv2D(64, (3,3), padding="same", strides=(1,1), use_bias=False)(inputs)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(64, (3,3), padding="same", strides=(1,1), use_bias=False)(conv1_1)
    conv1_2 = Activation('relu')(conv1_2)
    
    maxpool_1 = MaxPool2D(pool_size=(2,2))(conv1_2) #### 第一次pool 大小变成112*112
    
    conv2_1 = Conv2D(128, (3,3), padding="same", strides=(1,1), use_bias=False)(maxpool_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(128, (3,3), padding="same", strides=(1,1), use_bias=False)(conv2_1)
    conv2_2 = Activation('relu')(conv2_2)
    
    maxpool_2 = MaxPool2D(pool_size=(2,2))(conv2_2) #### 第二次pool 大小变成56*56
    
    conv3_1 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(maxpool_2)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(conv3_1)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(conv3_2)
    conv3_3 = Activation('relu')(conv3_3)
    
    maxpool_3 = MaxPool2D(pool_size=(2,2))(conv3_3) #### 第三次pool 大小变成28*28

    conv4_1 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(maxpool_3)
    conv4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(conv4_1)
    conv4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(256, (3,3), padding="same", strides=(1,1), use_bias=False)(conv4_2)
    conv4_3 = Activation('relu')(conv4_3)
    
    maxpool_4 = MaxPool2D(pool_size=(2,2))(conv4_3) #### 第四次pool 大小变成14*14

    flatten = Flatten()(maxpool_3)

    dense_1 = Dense(4096, activation='relu')(flatten)  #### 也可以把激活函数拆出来写成上面的那种 Activation('relu') 的格式
    dense_2 = Dense(1000, activation='relu')(dense_1)
    outputs = Dense(NUM_CLASSES, activation='softmax')(dense_2) ####　最后一定是和类别数相同的

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model    

#### 终端运行 python 文件名.py -p ???或者python 文件名.py -plot ??? 的话 ??? 就会传给args["plot"]
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--plot", default="plot.png",
                    help="path to loss and acc plot")
    args = vars(ap.parse_args()) ### vars就是为了方便管理变量，类似于把他们转变成了字典类的对象，可以通过名字调用 当ap很多很多个加了进来，就很方便
    return args

#### 训练过程
def classifier_train():    
    
    print("enter train process")
    
    path_traindatas = os.path.join('data','train')                                      #### 存放训练数据的路径
    path_validdatas = os.path.join('data','valid')                                      #### 存放验证集数据路径

    gen = MyGenerator(path_traindatas,path_validdatas, PATCH_SIZE,BATCH_SIZE,classnames) #### 每一批数据打包成一个yield放到显存中，
    model = Network()                                   #### 网络结构
    
    #### ** 返回x的y次幂 让学习率可以随代数的增加逐渐变小，得以得到最优解
    def schedule(epoch, decay=0.9):
        return LR * decay**(epoch)
    
    #### 确定什么时候保留模型 详情参见keras中文文档
    callbacks = [keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]
    
    #### 选择优化函数
    optim = keras.optimizers.Adam(lr=LR)
    # optim = keras.optimizers.RMSprop(lr=LR)
    # optim = keras.optimizers.SGD(lr=LR, momentum=0.9, decay= lrate / epochs, nesterov=False)
    
    #### 选择loss函数
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])        #### 二分类
    # model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy']) #### 多分类
    
    #### 每一代生成的模型相关数据就存在history这个量中
    #### fit_generator是keras自己的一个送数据的方式 每一次会产生一个yield 
    history = model.fit_generator(gen.generate(True), 
                                  steps_per_epoch = gen.train_batches,
                                  epochs = EPOCHES, verbose=1,                              
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.valid_batches,
                                  callbacks=callbacks)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHES), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHES), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHES), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHES), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])  

#### 主函数 只会在 python 文件名.py 时被调用 被import时不会执行
if __name__ == '__main__':
    starttime = datetime.now()    
    classifier_train()
    endtime = datetime.now()
    print("runtime:  ",(endtime-starttime).seconds)

    #### 跟时间有关的还有time 和 calendar（只能返回到 天 这个时间单位） 
    # import time
    # starttime = time.time()
    # time.sleep(1) #停顿1秒钟
    # endtime = time.time()
    # print(endtime-starttime)

  
