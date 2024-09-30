# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("-p", "--path", type=str, default="/data/yinyao/Projects/data/face_dataset/CelebA/Align_crop/", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_zp0.5_vstep5_gl5e-05_dl1e-05_", help="data path")
parser.add_argument("-mp", "--load_model_path", type=str, default="pretrain", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="l2svm_results/l2u_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_ff1", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_finalF_male_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_final_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_zp0.5_vstep5_gl5e-05_dl5e-05_ft0.5_rt0.5_tdr0.99_tdf0.99", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_final_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_zp0.5_vstep5_gl5e-05_dl5e-05_ft0.5_rt0.6", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_pretrainTrue_u1_s0.5_r2.5_i0.0_w0_gp150_im256_ss1000_ni30000_bs4_ug0.5_v0_inter0.0_stopTrue_zeroTrue", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_pretrainTrue_u1_s0.5_r2.5_i0.0_gp150_im256_ss100_ni30000_bs4_ug0.5_stopFalse_zp0.75", help="data path")
#parser.add_argument("-mp", "--load_model_path", type=str, default="final_results/l2svm_celebahq_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_zp0.5_vstep5_gl5e-05_dl1e-05", help="data path")
parser.add_argument("-p", "--path", type=str, default="/home/yinyao/Data/Projects/data/face_dataset/CelebA-HQ/CelebAMask-HQ/CelebA-HQ-img", help="data path")
parser.add_argument("-d", "--device", type=str, default='0', help="gpu device")
parser.add_argument("-g", "--growth", type=bool, default=False, help="allow_growth")
parser.add_argument("-s", "--step", type=int, default=0, help="train_step")
parser.add_argument("-ss", "--save_iter_step", type=int, default=1000, help="train_step")
parser.add_argument("-dl", "--d_lr", type=float, default=1e-5)
parser.add_argument("-gl", "--g_lr", type=float, default=5e-5)
parser.add_argument("-b1", "--beta1", type=float, default=0.5)
parser.add_argument("-b2", "--beta2", type=float, default=0.999)
parser.add_argument("-batch", "--batch_size", type=int, default=4)
parser.add_argument("-sample", "--sample_size", type=int, default=2)
parser.add_argument("-iters", "--iterations", type=int, default=100000)
parser.add_argument("-l1", "--lambda_u", type=float, default=1)
parser.add_argument("-l1g", "--lambda_ug", type=float, default=0.5)
parser.add_argument("-l2", "--lambda_s", type=float, default=0.5)
parser.add_argument("-l3", "--lambda_r", type=float, default=2.5)
parser.add_argument("-l4", "--lambda_i", type=float, default=0)
parser.add_argument("-l5", "--lambda_ff", type=float, default=1)
parser.add_argument("-gp", "--lambda_gp", type=float, default=150)
parser.add_argument("-img", "--img_size", type=int, default=256)
parser.add_argument("-v", "--vec_size", type=int, default=1)
parser.add_argument("-i", "--num_imgs", type=int, default=30000)
parser.add_argument("-wz", "--with_zero", type=bool, default=True)
parser.add_argument("-pre", "--pretrain", type=bool, default=True)
parser.add_argument("-zp", "--zero_p", type=float, default=0.5)
parser.add_argument("-vstep", "--v_step", type=int, default=5)
parser.add_argument("-ft", "--fake_t", type=float, default=0.3)
parser.add_argument("-rt", "--real_t", type=float, default=0.6)
parser.add_argument("-tds", "--t_decay_step", type=int, default=1000)
parser.add_argument("-tdr", "--t_decay_real", type=float, default=0.95)
parser.add_argument("-tdf", "--t_decay_fake", type=float, default=0.9)

args = parser.parse_args()

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from relgan_F import Relgan

# K.set_floatx('float64')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print(1, args.pretrain)

rel_gan = Relgan(args)
rel_gan.train()

