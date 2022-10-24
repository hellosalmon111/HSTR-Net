import numpy as np
import random
from PIL import Image
import os
import re
import torch
import cv2
import sys
import pickle
import pdb

patt = re.compile('\d+')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_transform(path):
    return pil_loader(path)

def key(x):
    lenth = len(x.split('_')[-1])
    return x[:-lenth], int(x[-lenth:-4])
def get_imglist(img_path):
    img_list = os.listdir(img_path)
    img_list.sort(key=key)
    path_list = []
    frame = {}
    i = 0
    for path in img_list:
        ss = os.path.splitext(path)[1]
        if ss == '.jpg':
            frame[path] = i
            path_list.append(path)
            i += 1
    return path_list, frame

def get_imgdic(img_path, loader):
    img_list = os.listdir(img_path)
    img_dic = {}
    i = 0
    for path in img_list:
        ss = os.path.splitext(path)[1]
        if ss == '.jpg':
            img_dic[path] = loader(img_path + path)
            i += 1
        if i % 1000 == 0:
            print(i)
    return img_dic

def get_img(img_path):
    img_list = os.listdir(img_path)
    img_dic = {}
    i = 0
    for path in img_list:
        ss = os.path.splitext(path)[1]
        if ss == '.jpg':
            img_dic[path] = default_transform(img_path + path)
            i += 1
        if i % 1000 == 0:
            print(i)
    return img_dic

def get_idx(img_path, name_list, img_name, frames, num_sequence):

    name, num = key(img_name)
    idx = frames[img_name]

    idxs = list(range(idx-num_sequence+1, idx+1))
    new_idx = [img_name] * num_sequence

    i = 0
    for id in idxs:

        new_img_name = name_list[id]
        new_name, nwe_num = key(new_img_name)
        if id < 0 or new_name != name:
            continue
        new_idx[i] = new_img_name
        i += 1

    return new_idx

class ImagesList(object):
    def __init__(self, crop_size, path, img_path, images_lenth, img_dic, NUM_CLASS=12, phase='train', transform=None, target_transform=None,
                 loader=default_transform):

        image_list = open(path).readlines()
        self.imgs = []
        self.real_image = []
        self.path_idx = {}
        i = 0
        self.img_dic = img_dic
        for f in image_list:
            fname, flabel, fpos = f.split('->')
            if fname in self.img_dic:
                self.imgs.append(f)
        
        self.name_list, self.st_frame = get_imglist(img_path)
        self.images_lenth = images_lenth
        self.img_path = img_path
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase
        self.NUM_CLASS = NUM_CLASS

    def __getitem__(self, index):
        f = self.imgs[index]
        fname, flabel, fpos = f.split('->')

        images = []
        images_path = get_idx(self.img_path, self.name_list, fname, self.st_frame, self.images_lenth)

        for path in images_path:
            dataps = np.zeros((12, 4))
            im_name = path.split('/')[-1]
            img = self.img_dic[path]
            if self.phase == 'train':
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)

                if self.transform is not None:
                    img = self.transform(img, flip, offset_x, offset_y)
            else:
                w, h = img.size
                offset_y = (h - self.crop_size) / 2
                offset_x = (w - self.crop_size) / 2
                if self.transform is not None:
                    img = self.transform(img)

            images.append(img.unsqueeze(0))

        au = np.array(patt.findall(flabel)).astype(int)
        au = torch.from_numpy(au.astype('float32'))
        return torch.cat(images, 0), au

    def __len__(self):
        return len(self.imgs)
