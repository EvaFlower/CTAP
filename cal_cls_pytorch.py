"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import keras
from keras.models import load_model

import numpy as np
import resnet

from read_lmdb import *

import os
import sys
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from skimage import io, transform
from keras.backend.tensorflow_backend import set_session
from contrib.ops import SwitchNormalization
from module import *
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle
import csv
import torch

import argparse
from attr_clf.src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from attr_clf.src.model import Classifier
from attr_clf.src.utils import get_optimizer, reload_model, print_accuracies
from attr_clf.src.loader import load_celebahq_images, DataSampler
from attr_clf.src.evaluation import compute_accuracy

from utils.dcgan_utils import save_images
import scipy
from PIL import Image

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument("--name", type=str, default="default",
                    help="Experiment name")
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3,
                    help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Smiling",
                    help="Attributes to classify")
parser.add_argument("--attr_name", type=str, default="smiling",
                    help="Attributes to classify")
parser.add_argument("--init_fm", type=int, default=32,
                    help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512,
                    help="Number maximum of filters in the autoencoder")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension")
parser.add_argument("--v_flip", type=bool_flag, default=False,
                    help="Random vertical flip for data augmentation")
parser.add_argument("--h_flip", type=bool_flag, default=True,
                    help="Random horizontal flip for data augmentation")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="Classifier optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradient norms (0 to disable)")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=27000,
                    help="Number of samples per epoch")
parser.add_argument("--reload", type=str, default="",
                    help="Reload a pretrained classifier")
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only load a subset of the whole dataset)")
params = parser.parse_args()

params.n_attr = 2 #ae.n_attr


attr_name = params.attr_name
if attr_name == 'mouth':
    params.attr = [('Mouth_Slightly_Open', 2)]
elif attr_name == 'High_Cheekbones':
    params.attr = [('High_Cheekbones', 2)]
else:
    params.attr = [(attr_name.capitalize(), 2)]
model_path = 'attr_clf/celebahq_{}_best.pth'.format(attr_name)
classifier = Classifier(params).cuda()
#f params.reload:
reload_model(classifier, model_path,
                 ['img_sz', 'img_fm', 'init_fm', 'hid_dim', 'attr', 'n_attr'])
print(attr_name, params.attr)

#model_path = '../keras-resnet/model_2/model_028-0.9317.hdf5'

#nb_classes = 2

#model = load_model(model_path)
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])


# The data, shuffled and split between train and test sets:
num_imgs = 3000
ratio = 0.9

version_start = 0 
version_end = 100
version_step = 1 
run = 2
path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5/'
# path = 'final_results/l2svm_celebahq_finalF_male_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5/'
# path = 'final_results/l2svm_celebahq_finalF_mouth_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
# path = 'final_results/l2svm_celebahq_finalF_ii_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
# #path = 'final_results/l2svm_celebahq_finalF_High_Cheekbones_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
# path = 'final_results/l2svm_celebahq_finalF_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.0_stopFalse_gl5e-05_dl1e-05_zp0.5/'
# path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainFalse_u0.0_s0.5_r2.5_i0_gp150_im64_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
# path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.0_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue/'
# path = 'final_results/l2svm_celebahq_finalF_smiling_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_cls/'
path = 'final_results/l2svm_celebahq_finalF_{}_interG_pretrainTrue_u1_s0.5_r2.5_i0_gp150_im256_ss1000_ni30000_bs4_ug0.5_stopFalse_gl5e-05_dl1e-05_zp0.5_wzTrue_v2/{}/'.format(attr_name, run)
bs = 15

# load real imgs
num_train = int(num_imgs*ratio)
num_test = int(num_imgs*(1-ratio))
start = 27000 
attr_name = 'smiling'
imgIndex = np.load("imgIndex_{}.npy".format(attr_name), allow_pickle=True)[:-1]
imgAttr = np.load("anno_dic_{}.npy".format(attr_name), allow_pickle=True).item()
print(len(imgIndex), len(imgAttr))
imgs = read_celebahq_lmdb(num_imgs, start=start)
targets = [imgAttr[imgIndex[i]] for i in range(start, start+num_imgs)]
targets = np.array(targets)
print(imgs.shape, targets.shape)
X_test = imgs#[num_train:]
y_test = targets #[num_train:]
gen_ys = 1-y_test
vs = gen_ys-y_test
attrs = []
gen_ys = np.squeeze(gen_ys)
for name, n_cat in params.attr:
    for i in range(n_cat):
        attrs.append(torch.FloatTensor((gen_ys == i).astype(np.float32)))
gen_ys = torch.cat([x.unsqueeze(1) for x in attrs], 1)

#X_test = X_test/127.5-1
#gen_ys = np_utils.to_categorical(gen_ys, nb_classes)
# Convert class vectors to binary class matrices.

X_test = X_test.astype('float32')
num_gen = len(X_test) 

print(X_test.shape, y_test.shape, gen_ys.shape, vs.shape)
# generator
img_shape = (256, 256, 3)
vec_shape = (1,)

imgA_input = Input(shape=img_shape)
imgB_input = Input(shape=img_shape)
vec_input_pos = Input(shape=vec_shape)
vec_input_neg = Input(shape=vec_shape)

g_out = generator(imgA_input, vec_input_pos, 256)
relGan = Model(inputs=[imgA_input, vec_input_pos], outputs=g_out)

cls_acc = []
iters = []
for version in tqdm(range(version_start, version_end, version_step)):
    train_path = path+'model/generator'+str(version)+'.h5'
    relGan.load_weights(train_path)
    gen_imgs = []
    for i in tqdm(range(num_gen//bs)):
        img = X_test[bs*i: bs*(i+1)]
        img = img/127.5-1
        v = vs[bs*i:bs*(i+1)]
        v = np.reshape(v, (bs, 1))
        gen_img, _ = relGan.predict([img, v])
        #save_images(gen_img, (256, 256, 3), 'cls_pytorch.jpg')
        gen_img = (gen_img+1)*127.5
        gen_img = gen_img.astype(np.uint8)
        gen_imgs.append(gen_img)
    #print(gen_ys[-10:])
    gen_imgs = np.concatenate(gen_imgs, axis=0)
    #width = 10
    #height = 1
    #new_im = Image.new('RGB', (256*height, 256*width))
    #for ii in range(height):
    #    for jj in range(width, 0):
    #        image = gen_imgs[-jj]
    #        new_im.paste(Image.fromarray(image,"RGB"), (256*ii,256*jj))
    #new_im.save('cls_pytorch.jpg')
    gen_imgs = np.rollaxis(gen_imgs, 3, 1)  
    gen_imgs = torch.from_numpy(gen_imgs)
    import pdb; pdb.set_trace()
    test_data = DataSampler(gen_imgs, gen_ys, params)
    print(gen_imgs.shape, gen_ys.shape)
    test_accu = compute_accuracy(classifier, test_data, params)
    test_accu = np.mean(test_accu)
    print(test_accu)
    cls_acc.append(test_accu)
    iters.append(version)
filename = path+'cls_acc_pytorch.csv'
from csv_rw import *
lists = [iters, cls_acc]
lists = zip(*lists)
create_csv(filename, lists)

