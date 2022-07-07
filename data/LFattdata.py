from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms
# from epinet_fun.func_generate_traindata_noise import generate_traindata_for_train
# from epinet_fun.func_generate_traindata_noise import data_augmentation_for_train
# from epinet_fun.func_generate_traindata_noise import generate_traindata512

import numpy as np
import torch
from torch.utils.data import Dataset

import os
import imageio
from data.data_util import *
mytransforms = transforms.Compose([transforms.ToTensor()])

root = '/data/lipeng//HCI/'
stanford_root = '/data/lipeng/stanford/'
hciold_root = '/data/lipeng//HCIOLD/'


## hci_dataset
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
            dir_LFimages += \
            ['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard',\
            'test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
        self.traindata_all, self.traindata_label = load_hci(dir_LFimages)

        # print('Load hciold data...')
        # hciold_images=[
        #     'blender/medieval',  'blender/horses', 'blender/stillLife', # statue
        # 'blender/monasRoom', 'blender/papillon','blender/buddha', 'blender/buddha2'] 
        # self.hciold_all, self.hciold_label = load_hciold(hciold_images)

        # traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ = generate_traindata512(traindata_all,traindata_label, Setting02_AngualrViews) 
        print('Load training data... Complete')  

        # load invalid regions from training data (ex. reflective region)   
        boolmask_img4= imageio.imread(root + 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
        boolmask_img6= imageio.imread(root + 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
        boolmask_img15=imageio.imread(root + 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
        self.boolmask_img4  = 1.0*boolmask_img4[:,:,3]>0
        self.boolmask_img6  = 1.0*boolmask_img6[:,:,3]>0
        self.boolmask_img15 = 1.0*boolmask_img15[:,:,3]>0    

    def __len__(self):
        return 4000

    def __getitem__(self, index):
        # if index > 2000:
        return generate_hci_for_train(self.traindata_all, self.traindata_label,
                                                                self.input_size,self.label_size,1,
                                                                self.Setting02_AngualrViews,
                                                                self.boolmask_img4,self.boolmask_img6,self.boolmask_img15, self.use_v, self.inc)    
        # else:
        #     return generate_hciold_for_train(self.hciold_all, self.hciold_label, 64, 3)

# train_loader = torch.utils.data.DataLoader(
#     TrainDataset('train'),
#     batch_size=1,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=False)
# for data in train_loader:
#     print(0)

# hci
class ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size
       
        print('Load test data...') 
        # ## hci
        self.dir_LFimages_hci=['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard',
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami'] 
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

# hci old 
class ValDataset_(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hciold
        self.dir_LFimages=[
            'blender/medieval',  'blender/horses', 'blender/stillLife', # blender/statue
        'blender/monasRoom', 'blender/papillon', 'blender/buddha2'] #'blender/buddha',
        self.valdata_all, self.valdata_label = load_hciold(self.dir_LFimages)
        print('Load test data... Complete') 

    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self, index):
        """
        return:  0:input N*81*256*256  views=81
                 1:label N*256*256
                 2: name
        """
        ## hciold
        center = self.opt.use_views // 2
        name = self.dir_LFimages[index].split('/')  
        valid_data = self.valdata_all[index]
        h, w, _, _, _ = valid_data.shape
        sh, sw = (h-512)//2, (w-512)//2 
        valid_data = torch.FloatTensor(np.transpose(valid_data[sh:sh+512, sw:sw+512, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.valdata_label[index][sh:sh+512,sw:sw+512]) 
        return  valid_data, test_label, name[1]  # 81*512*512  512*512  array


class TestHciold(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hciold
        self.dir_LFimages=[
            'blender/medieval',  'blender/horses', 'blender/stillLife', # blender/statue
        'blender/monasRoom', 'blender/papillon', 'blender/buddha2', 'blender/buddha'] 
        self.valdata_all, self.valdata_label = load_hciold(self.dir_LFimages)
        print('Load test data... Complete') 

    def __len__(self):
        return len(self.dir_LFimages)*4

    def __getitem__(self, index):
        """
        return:  0:input N*81*256*256  views=81
                 1:label N*256*256
                 2: name
        """
        ## hciold
        patch_id = index % 4
        index = index//4
        center = self.opt.use_views // 2
        name = self.dir_LFimages[index].split('/')  
        valid_data = self.valdata_all[index]
        H, W, _, _, _ = valid_data.shape

        if patch_id==0:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:512, :512, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][:512,:512]) 
        elif patch_id==1:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][-512:, :512, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][-512:,:512]) 
        elif patch_id==2:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:512, -512:, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][:512,-512:])
        else:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][-512:, -512:, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][-512:,-512:])
        info = {'name': name[1], 'H':H, 'W':W, 'id':patch_id}
        return  valid_data, test_label, info # 81*512*512  512*512  array



class TestStanford(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hciold
        self.dir_LFimages=os.listdir(stanford_root)
        self.valdata_all, self.valdata_label = load_stanford_data(self.dir_LFimages)
        print('Load test data... Complete') 

    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self, index):
        """
        return:  0:input N*81*256*256  views=81
                 1:label N*256*256
                 2: name
        """
        ## hciold
        center = self.opt.use_views // 2
        name = self.dir_LFimages[index]
        valid_data = self.valdata_all[index]
        valid_data = torch.FloatTensor(np.transpose(valid_data[:, :, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.valdata_label[index]) 
        return  valid_data, test_label, name # 81*512*512  512*512  array
