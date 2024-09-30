"""
https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format
"""
import os
import os.path as osp
import sys
import glob
import pickle
import lmdb
import cv2
from lmdb_util import ProgressBar
import numpy as np
from skimage import transform

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
except ImportError:
    pass

def read_one_img(file_path, dim):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) #io.imread(filepath)
    #cv2.imwrite('ori.jpg', img)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    #cv2.imwrite('ori_256.jpg', img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_rgb = img_rgb/127.5-1
    #print(np.max(img_rgb), np.min(img_rgb))
    #image = (img_rgb/2+0.5)*255
    #image = image.astype(np.uint8)
    #img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('tmp.jpg', img_bgr)
    return img_rgb
    
# configurations
img_folder = '/home/yinyao/Data/Projects/data/face_dataset/CelebA-HQ/CelebAMask-HQ/CelebA-HQ-img'  # glob matching pattern
lmdb_save_path = '/home/yinyao/Data/Projects/data/face_dataset/CelebA-HQ/CelebAMask-HQ/CelebA-HQ_64.lmdb'
meta_info = {'name': 'CelebA-HQ'}
mode = 1  # 1 for reading all the images to memory and then writing to lmdb (more memory);
# 2 for reading several images and then writing to lmdb, loop over (less memory)
batch = 1000  # Used in mode 2. After batch images, lmdb commits.
###########################################
if not lmdb_save_path.endswith('.lmdb'):
    raise ValueError("lmdb_save_path must end with \'lmdb\'.")
#### whether the lmdb file exist
#if osp.exists(lmdb_save_path):
#    print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
#    sys.exit(1)
idx = 0
num_imgs = 30000
#shape = (256, 256)
shape = (64, 64)
img_list = [os.path.join(img_folder, str(i)+'.jpg') for i in range(num_imgs*idx, num_imgs*(idx+1))]
#print(img_list[10])
if mode == 1:
    print('Read images...')
    dataset = [read_one_img(v, shape) for v in img_list] 
    data_size = sum([img.nbytes for img in dataset])
elif mode == 2:
    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in img_list)
else:
    raise ValueError('mode should be 1 or 2')

key_l = []
resolution_l = []
print(data_size)
pbar = ProgressBar(len(img_list))
env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
txn = env.begin(write=True)  # txn is a Transaction object
for i, v in enumerate(img_list):
    pbar.update('Write {}'.format(v))
    base_name = osp.splitext(osp.basename(v))[0]
    key = base_name.encode('ascii')
    data = dataset[i] if mode == 1 else cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if data.ndim == 2:
        H, W = data.shape
        C = 1
    else:
        H, W, C = data.shape
    txn.put(key, data)
    key_l.append(base_name)
    resolution_l.append('{:d}_{:d}_{:d}'.format(C, H, W))
    # commit in mode 2
    if mode == 2 and i % batch == 1:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()

print('Finish writing lmdb.')

#### create meta information
# check whether all the images are the same size
same_resolution = (len(set(resolution_l)) <= 1)
if same_resolution:
    meta_info['resolution'] = [resolution_l[0]]
    meta_info['keys'] = key_l
    print('All images have the same resolution. Simplify the meta info...')
else:
    meta_info['resolution'] = resolution_l
    meta_info['keys'] = key_l
    print('Not all images have the same resolution. Save meta info for each image...')

#### pickle dump
pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
print('Finish creating lmdb meta info.')
