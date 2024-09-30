import numpy as np
import lmdb
import pickle
import os
from tqdm import tqdm


def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes

def read_celebahq_lmdb(num_imgs, start=0):
    start = start
    lmdb_save_path = '/home/yuanpan/Data/yinghua/CelebA-HQ_256.lmdb'
    env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False,
                    meminit=False)
    paths, sizes = _get_paths_from_lmdb(lmdb_save_path)
    imgs = []
    print(num_imgs)
    for index in tqdm(range(start, start+num_imgs)):
        resolution = [int(s) for s in sizes[index].split('_')]
        path = paths[index]
        img = _read_img_lmdb(env, path, resolution)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

#idx = 5
#import cv2
#cv2.imwrite('read.jpg', imgs[idx])
#print(imgs.shape, np.max(img), np.min(img))
#imgIndex = np.load("imgIndex_smiling.npy", allow_pickle=True)[:-1]
#imgAttr = np.load("anno_dic_smiling.npy", allow_pickle=True).item()
#print(len(imgIndex), len(imgAttr))
#targets = [imgAttr[imgIndex[i]] for i in range(num_imgs)]
#print(targets)
#print(sizes)

