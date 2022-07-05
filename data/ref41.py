from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from posix import listdir

from torch.utils.data.dataset import T
from torchvision import transforms
# from epinet_fun.func_generate_traindata_noise import generate_traindata_for_train
# from epinet_fun.func_generate_traindata_noise import data_augmentation_for_train
# from epinet_fun.func_generate_traindata_noise import generate_traindata512

import numpy as np
import torch
from torch.utils.data import Dataset

import os
import imageio
from tqdm import tqdm

# root = '/data/lipeng//light0.7/'

dtu_root = '/data/lipeng/ref41/'
mytransforms = transforms.Compose([transforms.ToTensor()])

def load_ref41(dir_LFimages):# 541*376
    traindata_all=np.zeros((len(dir_LFimages), 368, 512, 9, 9, 3),np.uint8)
    traindata_label=np.zeros((len(dir_LFimages),368, 512), np.uint8)
    image_id=0
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread(dtu_root+dir_LFimage+'/%d.png' % (i+1)))[4:372,14:526,:] # load LF images(9x9) 
            except:
                print(dtu_root+dir_LFimage+'/%d.png' % i)
            traindata_all[image_id,:,:,i//9,i-9*(i//9),:]=tmp  
            del tmp
        image_id=image_id+1
    return traindata_all, traindata_label


# dtuLF
class TrainDataset(Dataset):
    def __init__(self, opt, is_train=True, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = self.opt.input_size + 2*opt.pad
        self.label_size = self.input_size
        self.use_v = opt.use_views
        self.inc = opt.input_c
        print('Load training data...')
        dir_LFimages = listdir(dtu_root+'train_array/')
        self.traindata_all, self.traindata_label = load_ref41(['train_array/'+sub+'/' for sub in dir_LFimages])
        
        # traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ = generate_traindata512(traindata_all,traindata_label, Setting02_AngualrViews) 
        print('Load training data... Complete')  

      

    def __len__(self):
        return 4000

    def __getitem__(self, index):
        index = index // 80
        center = self.use_v // 2
        """ initialize image_stack & label """
        traindata_batch = np.zeros(
            ( self.input_size, self.input_size,  9, 9, self.inc),
            dtype=np.float32)

        traindata_batch_label = np.zeros((1, self.label_size, self.label_size))

        idx_start = np.random.randint(0, 368 - self.input_size)
        idy_start = np.random.randint(0, 512 - self.input_size)
        

        if self.inc == 1:
            traindata_batch= np.expand_dims(np.squeeze(
                0.299 * self.traindata_all[index:index + 1, idx_start: idx_start + self.input_size,
                    idy_start: idy_start + self.input_size, :, :, 0].astype(
                    'float32') +
                0.587 * self.traindata_all[index:index + 1, idx_start: idx_start + self.input_size,
                    idy_start: idy_start + self.input_size, :, :, 1].astype(
                    'float32') +
                0.114 * self.traindata_all[index:index + 1, idx_start: idx_start + self.input_size,
                    idy_start: idy_start + self.input_size, :, :, 2].astype(
                    'float32')), -1) 
        else:
            traindata_batch = np.squeeze(
                self.traindata_all[index:index + 1, idx_start: idx_start + self.input_size,
                    idy_start: idy_start + self.input_size, :, :, :].astype(
                    'float32'))


        traindata_batch_label[:, :] = self.traindata_label[index,
                                                                idx_start:idx_start  + self.label_size,
                                                                idy_start:idy_start +  self.label_size] / 255.


        traindata_batch = np.float32((1 / 255) * traindata_batch)
        

        traindata_all = torch.FloatTensor(np.transpose(traindata_batch[:, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))

        return traindata_all, torch.FloatTensor(traindata_batch_label)
        

class TestDataset(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
   
        print('Load test data...') 
        # self.dir_LFimages = listdir(dtu_root)[:5]
        self.dir_LFimages = ['test_array/IMG_1411_eslf/', 'test_array/IMG_1541_eslf/', 'test_array/IMG_1328_eslf/', 'test_array/IMG_1554_eslf/', 'test_array/IMG_1586_eslf/']

        self.valdata_all, self.valdata_label = load_ref41(self.dir_LFimages)
        if opt.input_c == 1 :
            self.valdata_all = np.expand_dims(self.valdata_all[:,:,:,:,:,0] * 0.299 + self.valdata_all[:,:,:,:,:,1] * 0.587 + self.valdata_all[:,:,:,:,:,2] * 0.114, -1)
        print('Load test data... Complete') 


    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self, index):
        """
        return:  0:input N*81*256*256  views=81
                 1:label N*256*256
                 2: name
        """
        center = self.opt.use_views // 2
        valid_data = self.valdata_all / 255.
        test_label = self.valdata_label / 255.
        name = self.dir_LFimages[index]
        # test_x = [torch.FloatTensor(np.transpose(valid_data[index, :,:, 4, j,:], (2,0,1))) for j in range(4-center, 5+center)]
        # test_y = [torch.FloatTensor(np.transpose(valid_data[index, :,:, j, 4,:], (2,0,1))) for j in range(4-center, 5+center)]
        # test_45 = [torch.FloatTensor(np.transpose(valid_data[index, :,:, j, j,:], (2,0,1))) for j in range(4-center, 5+center)]
        # test_135 = [torch.FloatTensor(np.transpose(valid_data[index, :,:, j, 8-j,:], (2,0,1))) for j in range(4-center, 5+center)]
        valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.valdata_label[index, :,:]).unsqueeze(0)
        return  valid_data,test_label, name  # 81*512*512  512*512  array