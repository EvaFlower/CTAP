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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

stat_path = './fid_stats_celebahq.npz'
version_start = 53 
version_end = 54
version_step = 1
num_gen = 30000
attr_name = 'smiling'
run = 2

path = 'l2svm_results/l2u_celebahq_finalFc_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_ff1/'
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r0.0_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r1.0_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5/'
path = 'final_results/l2svm_celebahq_finalF_male_interG_pretrainTrue_u1_s0.5_r1.0_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_mouth_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_Wearing_Lipstick_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_ii_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/' 
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u0.0_s0.5_r2.5_i0_gp150_im64_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
#path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.0_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_cls/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u1_s0.0_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/'
path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/'
path = 'final_results/l2svm_celebahq_finalF_{}_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/{}/'.format(attr_name, run)
#attr_name = 'High_Cheekbones'
#attr_name = 'Wearing_Lipstick'
bs = 15

# load real imgs
images = read_celebahq_lmdb(num_gen)
imgIndex = np.load("imgIndex_{}.npy".format(attr_name), allow_pickle=True)[:-1]
imgAttr = np.load("anno_dic_{}.npy".format(attr_name), allow_pickle=True).item()
#imgIndex = np.load("imgIndex_smiling.npy", allow_pickle=True)[:-1]
#imgAttr = np.load("anno_dic_smiling.npy", allow_pickle=True).item()
print(len(imgIndex), len(imgAttr))
targets = [imgAttr[imgIndex[i]] for i in range(num_gen)]
targets = np.array(targets)
vs = 1-targets-targets
print(images.shape, vs.shape, np.sum(vs==0))

# generator
img_shape = (256, 256, 3)
vec_shape = (1,)

imgA_input = Input(shape=img_shape)
imgB_input = Input(shape=img_shape)
vec_input_pos = Input(shape=vec_shape)
vec_input_neg = Input(shape=vec_shape)

g_out = generator(imgA_input, vec_input_pos, 256)
relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)

fid_scores = []
iters = []
for version in tqdm(range(version_start, version_end, version_step)):
    train_path = path+'model/generator'+str(version)+'.h5'
    relGan.load_weights(train_path)
    gen_imgs = []
    for i in tqdm(range(num_gen//bs)):
        v = vs[bs*i:bs*(i+1)]
        v = np.reshape(v, (bs, 1))
        img = images[bs*i: bs*(i+1)]  
        img = img/127.5-1
        gen_img, _ = relGan.predict([img, v])
        gen_img = (gen_img/2+0.5)*255
        gen_img = gen_img.astype(np.uint8)
        gen_imgs.append(gen_img)
    gen_imgs = np.concatenate(gen_imgs, axis=0)
    fid_score = fid.calculate_fid_given_files(gen_imgs, stat_path)
    print(fid_score)
    fid_scores.append(fid_score)
    iters.append(version)
filename = path+'fid_{}.csv'.format(attr_name)
lists = [iters, fid_scores]
lists = zip(*lists)
from csv_rw import *
create_csv(filename, lists)
