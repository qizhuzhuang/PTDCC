from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os
import glob
from collections import OrderedDict
import random
from torchvision import models

from loss_overlap import vgg_feature


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.train_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            #  _AANAP_
            if data_name == 'warp1' or data_name == 'warp2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
            elif data_name == 'mask1' or data_name == 'mask2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.png'))
                self.datas[data_name]['image'].sort()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        vgg = vgg.cuda()
        self.feature = vgg.features[:4] + vgg.features[5:9]

        # print(self.datas.keys())

    def __getitem__(self, index):
        # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        # warp1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2HSV)
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = warp1 / 255
        warp1 = np.transpose(warp1, [2, 0, 1])
        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        # warp2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2HSV)
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = warp2 / 255
        warp2 = np.transpose(warp2, [2, 0, 1])
        # load mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = mask1 / 255
        mask1 = np.transpose(mask1, [2, 0, 1])
        # load mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = mask2 / 255
        mask2 = np.transpose(mask2, [2, 0, 1])
        # convert to tensor
        warp1_tensor = torch.tensor(warp1).cuda()
        warp2_tensor = torch.tensor(warp2).cuda()
        mask1_tensor = torch.tensor(mask1).cuda()
        mask2_tensor = torch.tensor(mask2).cuda()

        vgg_diff = torch.sqrt(torch.mean((self.feature(warp1_tensor) - self.feature(warp2_tensor)) ** 2, dim=0))

        return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff

    def __len__(self):
        return len(self.datas['warp1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):

        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('\\')[-1]
            if data_name == 'warp1' or data_name == 'warp2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
            elif data_name == 'mask1' or data_name == 'mask2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.png'))
                self.datas[data_name]['image'].sort()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        vgg = vgg.cuda()
        self.feature = vgg.features[:4] + vgg.features[5:9]

        print(self.datas.keys())

    def __getitem__(self, index):
        # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        # warp1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2HSV)
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = warp1 / 255
        warp1 = np.transpose(warp1, [2, 0, 1])
        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        # warp2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2HSV)
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = warp2 / 255
        warp2 = np.transpose(warp2, [2, 0, 1])

        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = mask1 / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = mask2 / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1).cuda()
        warp2_tensor = torch.tensor(warp2).cuda()
        mask1_tensor = torch.tensor(mask1).cuda()
        mask2_tensor = torch.tensor(mask2).cuda()

        vgg_diff = torch.sqrt(torch.mean((self.feature(warp1_tensor) - self.feature(warp2_tensor)) ** 2, dim=0))

        return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, vgg_diff

    def __len__(self):

        return len(self.datas['warp1']['image'])
