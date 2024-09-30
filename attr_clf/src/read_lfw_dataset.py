import cv2
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import h5py
import mat73


def read_one_img(file_path, dim):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) #io.imread(filepath)
    if img is None:
        print(file_path)
    #cv2.imwrite('ori.jpg', img)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    #cv2.imwrite('ori_256.jpg', img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(np.max(img_rgb), np.min(img_rgb))
    #print(np.max(img_rgb), np.min(img_rgb))
    #image = (img_rgb/2+0.5)*255
    #image = image.astype(np.uint8)
    #img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('tmp.jpg', img_bgr)
    return img_rgb


def load_images(img_path_list, shape):
    root_image_path = '/home/yuanpan/Data/yinghua/lfw/'
    n = len(img_path_list)
    imgs = [read_one_img(root_image_path+str(img_path_list[i]).replace('\\', '/'), shape) \
        for i in tqdm(range(n))]
    print(len(imgs))
    return imgs


def load_lfw_data(attr='Smiling', img_shape=(256,256)):
    anno_file_path = '/home/yuanpan/Data/yinghua/lfw_att_40.mat'
    annos = mat73.loadmat(anno_file_path)
    img_path_list = annos['name']
    imgs = load_images(img_path_list, img_shape)
    imgs = np.array(imgs)
    attr_name = np.array(annos['AttrName'])
    attrs = np.array(annos['label'])
    attr_idx = np.where(attr_name==attr)[0]
    return imgs, attrs[:, attr_idx]
    

if __name__ == '__main__':
    imgs, attrs = load_lfw_data()
    print(imgs.shape, attrs.shape)

