
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
import matplotlib
matplotlib.use('Agg')

from read_lmdb import *
import fid
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

from skimage.measure import compare_ssim
from PIL import Image


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

version_start = 53 
version_end = 54
version_step = 1
num_imgs = 1
num_start = 27000
img_shape = (256, 256, 3)
vec_shape = (1,)
v_step = 0.2
v_start = -1 
v_end = 1+v_step

num_imgs = 3000
num_start = 27000 
bs = 100
bs = np.min([bs, num_imgs])
im_size = 256
attr_name = 'smiling'
run = 2

path = 'results/celebahq_attr_smiling/'
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5/'
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r1.0_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5/'
path = 'final_results/l2svm_celebahq_finalF_male_interG_pretrainTrue_u1_s0.5_r1.0_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_mouth_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_Wearing_Lipstick_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_ii_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u0.0_s0.5_r2.5_i0_gp150_im64_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
#path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.0_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
#path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_cls/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u1_s0.0_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/'
path = 'final_results/l2svm_celebahq_finalF_{}_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/{}/'.format(attr_name, run)
save_path = path

start = 27000
imgs = read_celebahq_lmdb(num_imgs, start=start)
print(imgs.shape)
X_test = imgs#[num_train:]
X_test = X_test/127.5-1

imgA_input = Input(shape=img_shape)
imgB_input = Input(shape=img_shape)
vec_input_pos = Input(shape=vec_shape)
vec_input_neg = Input(shape=vec_shape)

g_out = generator(imgA_input, vec_input_pos, 256)
relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)


inters = np.arange(v_start, v_end, v_step)
ssim_scores = []
iters = []
for version in tqdm(range(version_start, version_end, version_step)):
    train_path = path+'model/generator'+str(version)+'.h5'
    relGan.load_weights(train_path)
    gen_imgs = np.zeros((num_imgs, inters.shape[0], img_shape[0], img_shape[1], img_shape[2]))
    #gen_imgs[:, 0] = (X_test+1)/2
    for n in tqdm(range(inters.shape[0])):
        for i in tqdm(range(num_imgs//bs)):
            img1 = X_test[bs*i: bs*(i+1)]
            v = np.tile(inters[n], (bs, 1))
            v = np.reshape(v, (bs, 1))
            gen_img, _ = relGan.predict([img1, v])
            gen_img = (gen_img+1)/2
            gen_imgs[bs*i:bs*(i+1), n] = gen_img
    print(gen_imgs.shape, np.min(gen_imgs), np.max(gen_imgs))

    ssims = np.zeros((num_imgs, inters.shape[0]-1))
    for i in range(num_imgs):
        for j in range(inters.shape[0]-1):
            img1 = gen_imgs[i,j]
            img2 = gen_imgs[i,j+1]
            s = ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())
            ssims[i, j] = s
    print(ssims.shape, ssims[0])
    std_ssims = np.zeros(num_imgs)
    for i in range(num_imgs):
        std_ssims[i] = np.std(ssims[i])
    print(np.mean(std_ssims))
    ssim_scores.append(np.mean(std_ssims))
    iters.append(version)
    width = 1
    height = inters.shape[0]
    images = gen_imgs[2]
    new_im = Image.new('RGB', (im_size*height, im_size*width))
    for ii in range(height):
        for jj in range(width):
            index=ii*width+jj
            image = (images[index])*255
            image = image.astype(np.uint8)
            new_im.paste(Image.fromarray(image,"RGB"), (im_size*ii,im_size*jj))
    new_im.save(save_path+'inter_samples_-1to1.jpg'.format(0))

filename = path+'ssim_-1to1.csv'
lists = [iters, ssim_scores]
lists = zip(*lists)
from csv_rw import *
create_csv(filename, lists)

