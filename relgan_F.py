# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import random
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session
from module import *
from ops import *
from skimage import io, transform
from tensorboardX import SummaryWriter
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from read_lmdb import *
from keras.utils import np_utils


class Relgan():
    
    def __init__(self, args):
        
        self.path = args.path
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr
        self.b1 = args.beta1
        self.b2 = args.beta2
        self.batch = args.batch_size
        self.sample = args.sample_size
        self.iters = args.iterations
        self.lambda_u = args.lambda_u
        self.lambda_u_ = args.lambda_u
        self.lambda_ug = args.lambda_ug
        self.lambda_s = args.lambda_s
        self.lambda_r = args.lambda_r
        self.lambda_i = args.lambda_i
        #self.lambda5 = args.lambda5
        self.gp_l = args.lambda_gp
        self.decay = 0. #self.d_lr/self.iters
        self.imgSize = args.img_size
        self.sampleSize = args.img_size
        self.vecSize = args.vec_size
        self.num_imgs = args.num_imgs
        self.save_iter_step = args.save_iter_step
        self.step = args.step*self.save_iter_step
        self.with_zero = args.with_zero
        self.load_model_path = args.load_model_path
        self.pretrain = args.pretrain
        print(args.pretrain, self.pretrain)
        self.zero_p = args.zero_p
        self.v_step = args.v_step
        self.fake_t_e = args.fake_t
        self.real_t_e = args.real_t
        self.fake_t = float('inf')#args.fake_t #1.
        self.real_t = float('inf')#args.real_t #1.
        self.t_decay_real = args.t_decay_real
        self.t_decay_fake = args.t_decay_fake
        self.attr_name = 'smiling'#'High_Cheekbones' #'Wavy_Hair'
        run = 1
        self.result_path = 'results/celebahq_{}_pretrain{}_u{}_s{}_r{}/{}'.\
            format(self.attr_name, self.pretrain, self.lambda_u, self.lambda_s, self.lambda_r, run)
        self.sample_path = os.path.join(self.result_path, 'img')
        self.model_path = os.path.join(self.result_path, 'model')
        self.log_path = os.path.join(self.result_path, 'log')
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
            os.makedirs(self.model_path)
            os.makedirs(self.log_path)        
        #self.d_lr -= self.decay * self.step
 
        self.img_shape = (self.imgSize, self.imgSize, 3)
        self.vec_shape = (self.vecSize,)
       
        imgs, atts = self.get_all_imgs() 
        self.num_train = int(self.num_imgs*0.9)
        self.num_val = int(self.num_imgs*0.1)
        self.t_decay_step = self.num_train//(2*self.batch)
        self.all_imgs, self.all_atts = imgs[:self.num_train], atts[:self.num_train] 
        self.val_imgs, self.val_atts = imgs[self.num_train:(self.num_train+self.num_val)],\
            atts[self.num_train:(self.num_train+self.num_val)]
        self.vs = 1-self.val_atts-self.val_atts
        self.val_gen_atts = np_utils.to_categorical(1-self.val_atts, 2)  # 2 means 0/1 two classes
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        self.datagen = ImageDataGenerator(horizontal_flip=True)
        self.writer = SummaryWriter(self.log_path)
    
    def get_model(self):
        
        self.imgA_input = Input(shape=self.img_shape)
        self.imgB_input = Input(shape=self.img_shape)
        self.vec_input_pos = Input(shape=self.vec_shape)
        self.vec_input_neg = Input(shape=self.vec_shape)
            
        g_out = generator(self.imgA_input, self.vec_input_pos, self.imgSize)

        self.g_model = Model(inputs=[self.imgA_input, self.vec_input_pos], outputs=g_out)

        d_out = discriminator(self.imgA_input, self.imgB_input, self.vec_input_pos, self.imgSize, self.vecSize)

        self.d_model = Model(inputs=[self.imgA_input, self.imgB_input, self.vec_input_pos], \
                             outputs=d_out)
        
        #print(self.g_model.summary())
        #print(self.d_model.summary())
        # load pretrained model lambda_u = 0
        dis_path = self.load_model_path+'/discriminator.h5'
        gen_path = self.load_model_path+'/generator.h5'
        if self.pretrain:
            self.d_model.load_weights(dis_path, by_name=True)
            self.g_model.load_weights(gen_path)
        #plot_model(self.g_model, to_file='g_model.png')
        #plot_model(self.d_model, to_file='d_model.png')
        
        # evaluate acc of translation
        # model_path = '../keras-resnet/model_2/model_028-0.9317.hdf5'
        # self.cls_model = load_model(model_path)
        # self.cls_model.compile(loss='categorical_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])


    def get_loss(self):
        
        def cal_df_gp():
            
            def cal_gp(gradients):
                
                gradients_sqr = K.square(gradients[0])
                gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
                gradient_l2_norm = K.sqrt(gradients_sqr_sum)
                gradient_penalty = K.mean(K.square(1 - gradient_l2_norm))
                return gradient_penalty
            
            alpha = K.random_uniform_variable(shape=(1,), low=0, high=1)
            
            mix_tar = alpha * self.img_a + (1 - alpha) * self.img_a2b
            
            mix_outputs_a2b = self.d_model([self.img_a, mix_tar, self.vec_ag])
            
            gradients_a2b = K.gradients([mix_outputs_a2b[0]], [mix_tar])
            
            df_gp = cal_gp(gradients_a2b) 
                
            return df_gp
        
        def lsgan(xs, ts):
            real = 0
            fake = 0
            x = K.reshape(xs[0], [self.batch, -1])
            t = K.reshape(ts[0], [self.batch, -1])
            real = K.mean(K.square(x-t))
            x = K.reshape(xs[1], [self.batch, -1])
            t = K.reshape(ts[1], [self.batch, -1])
            fake = K.mean(K.square(x-t))
            return real, fake
        
        def lsuggan(xs, ts, fake_weights=1):
            real = 0
            fake = 0
            x = K.reshape(xs[0], [self.batch, -1])
            t = K.reshape(ts[0], [self.batch, -1])
            real = K.mean(K.sum(K.square(x-t), axis=[-1]))
            x = K.reshape(xs[1], [self.batch, -1])
            t = K.reshape(ts[1], [self.batch, -1])
            fake = K.mean(K.sum(fake_weights*K.square(x-t), axis=[-1]))
            return real, fake
        
        self.img_a = Input(shape=self.img_shape)
        self.img_b = Input(shape=self.img_shape)

        self.vec_ab_pos = Input(shape=self.vec_shape)
        self.vec_ag = Input(shape=self.vec_shape)
        
        self.img_a2b, self.enc_a2b = self.g_model([self.img_a, self.vec_ag])
        self.img_a2a, self.enc_a2a = self.g_model([self.img_a, K.zeros_like(self.vec_ab_pos)])
        self.img_a2b2a, _ = self.g_model([self.img_a2b, -self.vec_ag])

        input_real = [self.img_a, self.img_b, self.vec_ab_pos]
        input_fake = [self.img_a, self.img_a2b, self.vec_ag]
 
        dc_real, d_real, _ = self.d_model(input_real)       
        dc_fake, d_fake, _ = self.d_model(input_fake)
   
        ones = K.ones_like(d_real)
        zeros = K.zeros_like(d_real)

        #fake_weight = K.less(K.abs(dc_fake), self.fake_t*K.ones_like(dc_fake))
        #fake_weight = K.cast(fake_weight, dtype='float32')
        #real_weight = K.less(self.real_t*K.ones_like(dc_fake), K.abs(dc_fake))
        #real_weight = K.cast(real_weight, dtype='float32')
        self.df_loss_real, self.df_loss_fake = lsgan([d_real, d_fake], [ones, zeros]) # GAN loss to keep quality
        self.dc_loss_real, self.dc_loss_fake = lsuggan([dc_real, dc_fake], [self.vec_ab_pos, zeros]) #, fake_weight)  # Universum loss to make attribute transfer
        #vec_ag = K.cast(K.less(0.1*K.ones_like(dc_fake), K.abs(dc_fake)), dtype='float32')*K.sign(self.vec_ag)
        #self.dc_loss_real, self.dc_loss_fake_2 = lsuggan([dc_real, dc_fake], [self.vec_ab_pos, vec_ag]) #, real_weight) #, 1-fake_weight)  # Universum loss to make attribute transfer
        #self.dc_loss_fake = (self.dc_loss_fake_1+self.dc_loss_fake_2)
        self.df_loss = 0.5*(self.df_loss_real+self.df_loss_fake)
        self.dc_loss = 0.5*(self.dc_loss_real+self.lambda_ug*self.dc_loss_fake)

        print('self.df_loss', K.int_shape(self.df_loss))
        print('self.dc_loss', K.int_shape(self.dc_loss))
        
        self.df_gp = cal_df_gp()
       
        self.d_loss = self.lambda_s*self.df_loss + self.lambda_u*self.dc_loss + self.gp_l*self.df_gp 
        
        self.gf_loss_real, self.gf_loss_fake = lsgan([d_real, d_fake], [zeros, ones])
        self.gc_loss_real, self.gc_loss_fake = lsuggan([dc_real, dc_fake], [zeros, self.vec_ag], ones)
        self.gf_loss = 0.5*(self.gf_loss_real+self.gf_loss_fake)
        self.gc_loss = 0.5*(self.gc_loss_real+self.gc_loss_fake)
        
        g_loss_rec1 = K.mean(K.abs(self.img_a - self.img_a2b2a))
        g_loss_rec2 = K.mean(K.abs(self.img_a - self.img_a2a)) #identity loss
        
        print('self.gf_loss', K.int_shape(self.gf_loss))
        print('self.gc_loss', K.int_shape(self.gc_loss))
        print('self.g_loss_rec1', K.int_shape(g_loss_rec1))
        print('self.g_loss_rec2', K.int_shape(g_loss_rec2))
        
        self.gr_loss = self.lambda_r * g_loss_rec1 + self.lambda_i * g_loss_rec2 
        self.g_loss = self.lambda_s*self.gf_loss + self.lambda_u*self.gc_loss + self.gr_loss 
        
    def get_optimizer(self):
        
        g_opt = Adam(lr=self.g_lr, decay = self.decay, beta_1=self.b1, beta_2=self.b2)
        g_weights = self.g_model.trainable_weights
        g_inputs = [self.img_a, self.img_b, self.vec_ab_pos, self.vec_ag]
        
        self.g_training_updates = g_opt.get_updates(g_weights, [], self.g_loss)
        self.g_train = K.function(g_inputs, 
                                  [K.mean(self.g_loss), 
                                   K.mean(self.gf_loss), 
                                   K.mean(self.gc_loss), 
                                   K.mean(self.gr_loss),
                                   K.mean(self.gf_loss_real),
                                   K.mean(self.gf_loss_fake),
                                   K.mean(self.gc_loss_real),
                                   K.mean(self.gc_loss_fake)],
                                   #K.mean(self.g_inter_loss),
                                   #K.mean(self.gi_loss)],
                                  self.g_training_updates)
        
        d_opt = Adam(lr=self.d_lr, decay = self.decay, beta_1=self.b1, beta_2=self.b2)
        d_weights = self.d_model.trainable_weights
        d_inputs = [self.img_a, self.img_b, self.vec_ab_pos, self.vec_ag]
        
        self.d_training_updates = d_opt.get_updates(d_weights, [], self.d_loss)
        self.d_train = K.function(d_inputs, 
                                  [K.mean(self.d_loss), 
                                   K.mean(self.df_loss), 
                                   K.mean(self.dc_loss), 
                                   K.mean(self.gp_l * self.df_gp),
                                   K.mean(self.df_loss_real),
                                   K.mean(self.df_loss_fake),
                                   K.mean(self.dc_loss_real),
                                   K.mean(self.dc_loss_fake)],
                                  self.d_training_updates)
       
    def get_all_imgs(self):
        num_imgs = self.num_imgs
        imgIndex = np.load("imgIndex_{}.npy".format(self.attr_name), allow_pickle=True)[:-1]
        imgAttr = np.load("anno_dic_{}.npy".format(self.attr_name), allow_pickle=True).item()
        #imgIndex = np.load("imgIndex_mouth.npy", allow_pickle=True)[:-1]
        #imgAttr = np.load("anno_dic_mouth.npy", allow_pickle=True).item()
        #imgIndex = np.load("imgIndex_smiling.npy", allow_pickle=True)[:-1]
        #imgAttr = np.load("anno_dic_smiling.npy", allow_pickle=True).item()
        print(len(imgIndex), len(imgAttr))
        imgs = read_celebahq_lmdb(num_imgs, im_size=self.imgSize)
        targets = [imgAttr[imgIndex[i]] for i in range(num_imgs)]
        targets = np.array(targets)
        print(imgs.shape, targets.shape)
        print(np.max(imgs), np.min(imgs), np.max(targets), np.min(targets))
        return imgs, targets

    def get_one_img(self, filepath):
        img = io.imread(filepath) 
        img = transform.resize(img, (self.imgSize, self.imgSize), preserve_range=True)
        img = img/127.5-1
        return img

    def get_imgs_tags(self, indexserX1, indexserX2):
        # remove pair with same attributes
        if self.with_zero==False:
            for i in range(self.batch):
                while self.all_atts[indexserX1[i]] == self.all_atts[indexserX2[i]]:
                    temp_index = np.random.choice(self.num_train, 1)[0]
                    indexserX2[i] = temp_index 
        #self.datagen.fit(imgs)
        #imgs = self.datagen.flow(imgs, batch_size=self.batch, shuffle=False).next()
        imgs = self.all_imgs[indexserX1]
        atts = self.all_atts[indexserX1]
        imgs_pair = self.all_imgs[indexserX2]
        atts_pair = self.all_atts[indexserX2]
        #print(np.max(self.all_imgs), np.min(self.all_imgs)) 
        imgs = imgs/127.5-1
        imgs_pair = imgs_pair/127.5-1
        imgs = np.array(imgs)
        atts = np.array(atts).reshape(-1, self.vecSize)
        imgs_pair = np.array(imgs_pair)
        atts_pair = np.array(atts_pair).reshape(-1, self.vecSize)
        
        return imgs, atts, imgs_pair, atts_pair
    
    def val_acc(self):
        bs = 15 
        gen_imgs = []
        for i in tqdm(range(self.num_val//bs)):
            img = self.val_imgs[bs*i: bs*(i+1)]
            img = img/127.5-1
            v = self.vs[bs*i:bs*(i+1)]
            v = np.reshape(v, (bs, 1))
            gen_img, _ = self.g_model.predict([img, v])
            gen_imgs.append(gen_img)
        gen_imgs = np.concatenate(gen_imgs, axis=0)
        print(gen_imgs.shape, self.val_gen_atts.shape)
        cls_acc = self.cls_model.evaluate(gen_imgs, self.val_gen_atts, verbose=0)[1]
        return cls_acc 
                  
    def train(self):
       
        print("training")
        
        ite = self.step
        
        num_train = self.num_train 
        def getIndex():
            while True:
                count = 0
                index_permutation = np.random.permutation(num_train)
                while count + self.batch*2 < num_train:
                    yield index_permutation[count:(count+self.batch*2)]
                    count = count + self.batch*2
        
        index_gen = getIndex()
        
        def get_training_data(wrong=False):
            indexser = next(index_gen)
            indexser1 = indexser[self.batch*0:self.batch*1] 
            indexser2 = indexser[self.batch*1:self.batch*2]

            img_as, att_as, img_bs, att_bs = self.get_imgs_tags(indexser1, indexser2)
            vec_ab_pos = att_bs - att_as
           
            p = [(1-self.zero_p)/2, self.zero_p, (1-self.zero_p)/2]
            vec_ag = np.random.choice([-1., 0,  1], self.batch, p) 
            vec_ag = np.reshape(vec_ag, (self.batch, self.vecSize))
            #a = np.random.randint(0, self.v_step+1, size=(self.batch, self.vecSize))/self.v_step
            a = np.random.uniform(-1.0, 1.01, size=(self.batch, self.vecSize))
            #a = np.random.uniform(-2.0, 2.01, size=(self.batch, self.vecSize))
            vec_ag = a #*vec_ag
            #vec_ag = np.random.choice([-1.,  1], (self.batch, self.vecSize)) 
            #a = np.random.uniform(0.01, 1, size=(self.batch, self.vecSize))
            #vec_ag = a*vec_ag
            return img_as, img_bs, vec_ab_pos, vec_ag

        self.best_cls_acc = -1 
        for ite in tqdm(range(int(self.iters))):
            t_start = time.time()
             
            weights = self.d_model.get_weights()[16]
            w = 0.5*np.mean(np.square(weights))
            img_as, img_bs, vec_ab_pos, vec_ag = get_training_data(wrong=False)
            for i in range(1):
                errD = self.d_train([img_as, img_bs, vec_ab_pos, vec_ag])
                
            for i in range(1):
                errG = self.g_train([img_as, img_bs, vec_ab_pos, vec_ag])
            
            t_end = time.time()
            
            self.writer.add_scalar('d_loss', errD[0], ite)
            self.writer.add_scalar('g_loss', errG[0], ite)
            self.writer.add_scalar('df_loss', errD[1], ite)
            self.writer.add_scalar('gf_loss', errG[1], ite)
            self.writer.add_scalar('dc_loss', errD[2], ite)
            self.writer.add_scalar('gc_loss', errG[2], ite)
            self.writer.add_scalar('gr_loss', errG[3], ite)
            #self.writer.add_scalar('inter_loss', errG[4], ite)
            self.writer.add_scalar('gp_loss', errD[3], ite)
            self.writer.add_scalar('df_loss_real', errD[4], ite)
            self.writer.add_scalar('df_loss_fake', errD[5], ite)
            self.writer.add_scalar('dc_loss_real', errD[6], ite)
            self.writer.add_scalar('dc_loss_fake', errD[7], ite)
            self.writer.add_scalar('gf_loss_real', errG[4], ite)
            self.writer.add_scalar('gf_loss_fake', errG[5], ite)
            self.writer.add_scalar('gc_loss_real', errG[6], ite)
            self.writer.add_scalar('gc_loss_fake', errG[7], ite)
            self.writer.add_scalar('_w_loss', w, ite)
            #self.writer.add_scalar('gi_loss', errG[5], ite)
            #self.writer.add_scalar('di_loss', errD[4], ite)
            
            #if ite>50000:
            #    self.fake_t = args.fake_t 
            #    self.real_t = args.real_t
            if ite%self.t_decay_step==0 and False:
                #print(self.fake_t, self.real_t)      
                self.fake_t = 1.*self.t_decay_fake**((ite-0)//self.t_decay_step)
                self.fake_t = np.max([self.fake_t_e, self.fake_t])
                self.real_t = 1.*self.t_decay_real**((ite-0)//self.t_decay_step)
                self.real_t = np.max([self.real_t_e, self.real_t])
            if ite%self.save_iter_step==0 or (ite+1)==self.iters:
                
                img_as, img_bs, vec_ab_pos, vec_ag = get_training_data(wrong=False)
                
                g_a2b = [img_as[:self.sample], vec_ag[:self.sample]]
                fakea2b,_ = self.g_model.predict(g_a2b)
                
                g_a2a = [img_as[:self.sample], np.zeros([self.sample, self.vecSize])]
                fakea2a,_ = self.g_model.predict(g_a2a)
                
                g_a2b2a = [fakea2b[:self.sample], -vec_ag[:self.sample]]
                fakea2b2a,_ = self.g_model.predict(g_a2b2a)
                
                images = np.concatenate([img_as[:self.sample], fakea2b, fakea2b2a, fakea2a], axis = 0)
                
                width = self.sample
                height = 4
                new_im = Image.new('RGB', (self.sampleSize*height, self.sampleSize*width))
                for ii in range(height):
                    for jj in range(width):
                        index=ii*width+jj
                        image = (images[index]/2+0.5)*255
                        image = transform.resize(image, (self.sampleSize, self.sampleSize), preserve_range = True)
#                         image = image*255
                        image = image.astype(np.uint8)
                        new_im.paste(Image.fromarray(image,"RGB"), (self.sampleSize*ii,self.sampleSize*jj))
                filename = self.sample_path+"/fakeFace%d.jpg"%(ite//self.save_iter_step)
                new_im.save(filename)
                
                try:
                    self.g_model.save(self.model_path+"/generator%d.h5"%(ite//self.save_iter_step))
                    self.d_model.save(self.model_path+"/discriminator%d.h5"%(ite//self.save_iter_step))
                except:
                    print('Pass save')
        



