import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class FRCNNDataset(Dataset):   #继承自Dataset类
    def __init__(self, train_lines, shape=[600,600], is_train=True):
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.shape = shape
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
        
    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()  #2007_train.txt的每一行的图片地址,后面的每一个都是五个数的字符串,代表(x1,y1,x2,y2,类别)
        image = Image.open(line[0])
        iw, ih = image.size
        h,w = self.shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        #将坐标变成整数,且放在二维列表中,box.shape[0]是ground truth的个数 ,box.shape[1]=5代表(x1,y1,x2,y2,label)
        if not random:   #如果是验证样本,则使用填充的方法resize图片
            # resize image    将图像等比例缩放,即长宽比不变,在与[600,600]的缺失位置填充128
            scale = min(w/iw, h/ih)  # 需要的尺寸/实际的尺寸  也即是将最大的边映射到对应的尺寸
            nw = int(iw*scale)      #resize后的图像尺寸
            nh = int(ih*scale)
            dx = (w-nw)//2         # resize后的图像尺寸与model需要的尺寸(600,600)之间的偏差的一半,将resize后的图像放在中间
            dy = (h-nh)//2

            #可以使用opencv+np.pad实现
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx   #更新box的(x1,x2)
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy   #更新box的(y1,y2)
                box[:, 0:2][box[:, 0:2]<0] = 0          #将更新后box尺寸限制在 [0,h],[0,w]之间
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] #限制ground truth的宽高至少为1
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        #如果是训练样本则进行数据增强
        # resize image
        new_ra = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter) #新的ratio
        scale = self.rand(.5, 1.5)  # resize图与[600,600]的比例
        if new_ra < 1: #说明 h>w
            nh = int(scale*h)
            nw = int(nh*new_ra)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ra)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data


    def __getitem__(self, index):
        img, y = self.get_random_data(self.train_lines[index], random=self.is_train) #img:一张图片,y:(x1,y1,x2,y2,label)
        img = np.transpose(img / 255.0, [2,0,1])    #图像归一化,并图像shape为(3,600,600),即通道在前(c,h,w)
        box = y[:, :4]
        label = y[:, -1]
        return img, box, label    #返回每张图像,ground truth 的box坐标,标签 都是列表

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels

