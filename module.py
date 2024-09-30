# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import functools
import tensorflow as tf
import keras
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import LeakyReLU, Activation, Input, Reshape, Flatten, Dense, Multiply
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, Lambda, Concatenate, Add, Subtract
from keras.layers import BatchNormalization, Dropout, Subtract, GlobalAveragePooling2D
from contrib.ops import SwitchNormalization
from ops import *


init_weight = 'he_normal'
res_init_weight = 'he_normal'
img_size = 256 
regular = None

def tileAttr(x):
        x = tf.expand_dims(x, axis = 1)
        x = tf.expand_dims(x, axis = 2)
        return tf.tile(x, [1, img_size, img_size, 1])
    
def tileAttr2(x):
        x = tf.expand_dims(x, axis = 1)
        x = tf.expand_dims(x, axis = 2)
        dim = img_size//64
        return tf.tile(x, [1,dim,dim,1]) #tf.tile(x, [1, 4, 4, 1])

def generator(img, attr, size):
    
    concat = Concatenate()([img, Lambda(tileAttr)(attr)])
    
    DownSample = functools.partial(Conv2D, padding="same" , kernel_initializer=init_weight, kernel_regularizer = orthogonal)
    UpSample = functools.partial(Conv2DTranspose, padding="same" , kernel_initializer=init_weight, kernel_regularizer = orthogonal)
    
    conv_in = DownSample(64, 7, name="conv_in_conv")(concat)
    conv_in = SwitchNormalization(axis=-1, name="conv_in_norm")(conv_in)
    conv_in = Activation('relu', name="conv_in_relu")(conv_in)
    
    down1 = DownSample(128, 4, strides=2, name="down1_conv")(conv_in)
    down1 = SwitchNormalization(axis=-1, name="down1_norm")(down1)
    down1 = Activation('relu', name="down1_relu")(down1)
    
    down2 = DownSample(256, 4, strides=2, name="down2_conv")(down1)
    down2 = SwitchNormalization(axis=-1, name="down2_norm")(down2)
    down2 = Activation('relu', name="down2_relu")(down2)
    
    resb = residual_block(down2, 256, 3, res_init_weight, 'block1')
    resb = residual_block(resb, 256, 3, res_init_weight, 'block2')
    resb = residual_block(resb, 256, 3, res_init_weight, 'block3')
    
    encode_out = resb
    
    resb = residual_block(resb, 256, 3, res_init_weight, 'block4')
    resb = residual_block(resb, 256, 3, res_init_weight, 'block5')
    resb = residual_block(resb, 256, 3, res_init_weight, 'block6')
    
    up2 = UpSample(128, 4, strides=2, name="up2_deconv2")(resb)
    up2 = SwitchNormalization(axis=-1, name="up2_norm")(up2)
    up2 = Activation('relu', name="up2_relu")(up2)
    brid2 = up2
    
    up1 = UpSample(64 , 4, strides=2, name="up1_deconv2")(brid2)
    up1 = SwitchNormalization(axis=-1, name="up1_norm")(up1)
    up1 = Activation('relu', name="up1_relu")(up1)
    brid3 = up1
    
    conv_out = DownSample(3, 7, name="conv_out_conv")(brid3)
    conv_out = Activation('tanh', name="conv_out_tanh")(conv_out)
    return conv_out, encode_out

def discriminator(imgA, imgB, attr, size, att_size):
    
    filters = [64, 128, 256, 512, 1024, 2048]
    
    convs = [Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv1"),
             Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv2"),
             Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv3"),
             Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv4"),
             Conv2D(1024, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv5"),
             Conv2D(2048, 4, strides=2, padding='same', kernel_initializer=init_weight, kernel_regularizer=regular, name="conv6")]
    
    
    #original image
    
    y1 = imgA
    for i in range(6):
        y1 = convs[i](y1)
        y11 = y1
        y1 = LeakyReLU(alpha=0.01)(y1)
    
    #target image
    y2 = imgB
    for i in range(6):
        y2 = convs[i](y2)
        y22 = y2
        y2 = LeakyReLU(alpha=0.01)(y2)
    
    #d_out_y1 = Conv2D(1, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regular)(y1)
    #d_out_y2 = Conv2D(1, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regular)(y2)
    d_out_y2 = Flatten()(y2)
    d_out_y2_f = d_out_y2
    d_out_y2 = Dense(1, kernel_initializer='lecun_normal', kernel_regularizer=regular)(d_out_y2)
    #d_out3 = Conv2D(64, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regular)(y2)
    #d_out3 = Lambda(lambda x: K.mean(x, axis=[-1]))(d_out3)
    
    d_out_y12_diff = Subtract()([y2, y1]) #Concatenate()([y1, y2, Lambda(tileAttr2)(attr)])  #
    d_out_y12 = Conv2D(2048, 1, strides=1, kernel_initializer='lecun_normal', kernel_regularizer=regular)(d_out_y12_diff)
    d_out_y12_diff_output = Subtract()([y22, y11]) 
    d_out_y12 = LeakyReLU(alpha=0.01)(d_out_y12) # 2 2 2048
    d_out_y12 = Flatten()(d_out_y12)
    d_out_y12 = Dense(att_size, kernel_initializer='lecun_normal', kernel_regularizer=regular)(d_out_y12)
    #d_out_y12 = Conv2D(1, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regular)(d_out_y12)
   
    #d_out_y2_inter = Conv2D(64, 1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regular)(y2)
    #d_out_y2_inter = Lambda(lambda x: K.mean(x, axis=[-1]))(d_out_y2_inter)
 
    return d_out_y12, d_out_y2, d_out_y12_diff_output #, d_out_y2_inter
