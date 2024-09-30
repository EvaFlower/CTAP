import os
import sys
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from skimage import io, transform
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from contrib.ops import SwitchNormalization
from module import *
import argparse
from utils.dcgan_utils import get_image
import pandas as pd
from read_lmdb import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


#path = 'results/l2svm_celebahq_with0_pretrainu0_u0.0_s1_r10_i0_w0_gp150_im256_ss1000_ni30000_bs4_ug1/'
path = 'final_results/l2svm_celebahq_pretrainTrue_u1_s0.5_r2.5_i0.0_w0_gp150_im256_ss1000_ni30000_bs4_ug0.5_v0_inter0.0_stopTrue_zeroTrue/'
img_shape = (256, 256, 3)
#vec_shape = (17,)
#img_shape = (64, 64, 3)
vec_shape = (1,)
im_size = 256

imgA_input = Input(shape=img_shape)
imgB_input = Input(shape=img_shape)
vec_input_pos = Input(shape=vec_shape)
vec_input_neg = Input(shape=vec_shape)

num_imgs = 3000 
images = read_celebahq_lmdb(num_imgs, start=30000-num_imgs)[999]
images = images/127.5-1
g_out = generator(imgA_input, vec_input_pos, im_size)
relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)

version = 1 
train_path = path+'model/generator'+str(version)+'.h5'
relGan.load_weights(train_path)
label = 0 
bs = 1
v = np.ones([1])*label
v = np.reshape(v, (bs, 1))
images = np.array(images)
images = np.reshape(images, (1, 256, 256, 3))
gen_img, _ = relGan.predict([images, v])
gen_img = (gen_img/2+0.5)*255
gen_img = gen_img.astype(np.uint8).reshape(img_shape)
cv2.imwrite(path+'translation_{}.jpg'.format(label), cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))    

