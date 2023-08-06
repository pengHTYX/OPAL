from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms

import numpy as np
import torch
from torch.utils.data import Dataset

import os
import imageio
from data.data_util import *
mytransforms = transforms.Compose([transforms.ToTensor()])

root = './dataset/hci_dataset/'

class TrainDataset(Dataset):
    def __init__(self, opt, istrain=True):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = 64
        self.label_size = self.input_size
        self.use_v = 9
        self.inc = 3
        print('Load hci data...')
        dir_LFimages = [
            'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
            'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
            'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
            'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl',]
        
        with_valid = True
        if with_valid:
            dir_LFimages += ['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
            # 'test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
        self.traindata_all, self.traindata_label = load_hci(dir_LFimages)

        print('Load training data... Complete')  

        # load invalid regions from training data (ex. reflective region)   
        boolmask_img4= imageio.imread(root + 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
        boolmask_img6= imageio.imread(root + 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
        boolmask_img15=imageio.imread(root + 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
        self.boolmask_img4  = 1.0*boolmask_img4[:,:,3]>0
        self.boolmask_img6  = 1.0*boolmask_img6[:,:,3]>0
        self.boolmask_img15 = 1.0*boolmask_img15[:,:,3]>0

        self.num_img = len(dir_LFimages)

    def __len__(self):
        return 4000

    def __getitem__(self, index):
        # if index > 2000:
        return generate_hci_for_train(self.traindata_all, self.traindata_label,
                                                                self.input_size,self.label_size,1,
                                                                self.Setting02_AngualrViews,
                                                                self.boolmask_img4,self.boolmask_img6,self.boolmask_img15, self.use_v, self.inc, self.num_img)  

# hci
class ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size
       
        print('Load test data...') 
        # ## hci
        self.dir_LFimages_hci=['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard'] 
        self.valdata_hci, self.label_hci = load_hci(self.dir_LFimages_hci)
        
        if opt.input_c == 1:
            self.valdata_hci = np.expand_dims(self.valdata_hci[:,:,:,:,:,0] * 0.299 + self.valdata_hci[:,:,:,:,:,1] * 0.587 + self.valdata_hci[:,:,:,:,:,2] * 0.114, -1)
        print('Load test data... Complete') 

    def __len__(self):
        return len(self.dir_LFimages_hci) 

    def __getitem__(self, index):
        center = self.opt.use_views // 2
        valid_data = self.valdata_hci / 255.
        name = self.dir_LFimages_hci[index].split('/')    
        valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.label_hci[index, :,:]) 
        return  valid_data, test_label, name[1]  # 81*512*512  512*512  array

