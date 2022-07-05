from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from posix import listdir

from torch.utils.data.dataset import T
from torchvision import transforms

from imageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import imageio
from scipy.io import loadmat, savemat
from .mytransforms import get_params, get_tranforms

root = '/data/lipeng/hci_dataset/'
mytransforms = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3), transforms.ToTensor()])

# 产生特定数据集 size=128 use_views=9

'''
class TrainDataset(Dataset):
    def __init__(self, opt, is_train=True, transform=None):
        self.opt = opt
        if is_train:
            self.root = 'dataset'
            self.w, self.h = 128, 128
            self.subpath = [os.path.join(self.root, sub) for sub in ['degree0', 'degree90', 'degree45', 'degree_45', 'label' ]]
        else:
            self.root = 'lytro'
            self.w, self.h = 552, 383
        self.num_view = 9
        self.AngualrViews = [i for i in range(self.num_view)]
        self.img_scale =  1 #   1 for small_baseline(default) <3.5px, 
                # 0.5 for large_baseline images   <  7px
        self.use_views = 9
        self.key = ['d0', 'd90', 'd45', 'd_45', 'disparity']
        self.db = self._get_db() 
    
    def _get_db(self):
        db = []
        
        names = listdir(self.subpath[-1])
        for name in names:
            item = {}
            for k, p in enumerate(self.subpath):
                data = loadmat(os.path.join(p, name))['data']
                item[self.key[k]] = data
            db.append(item)
        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        item = self.db[index]
        d0 = item[self.key[0]]
        d0 = torch.from_numpy(np.transpose(d0, (2, 0, 1)))
        d1 = item[self.key[1]]
        d1 = torch.from_numpy(np.transpose(d1, (2, 0, 1)))
        d2 = item[self.key[2]]
        d2 = torch.from_numpy(np.transpose(d2, (2, 0, 1)))
        d3 = item[self.key[3]]
        d3 = torch.from_numpy(np.transpose(d3, (2, 0, 1)))

        label = item[self.key[4]]
        label = torch.from_numpy(label).unsqueeze(0)
        return d0, d1, d2, d3, label
    
class TestDataset(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        if not is_lytro:
            self.root = [
            'hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots', 'hci_dataset/stratified/pyramids', 'hci_dataset/stratified/stripes',
            'hci_dataset/training/boxes', 'hci_dataset/training/cotton', 'hci_dataset/training/dino', 'hci_dataset/training/sideboard']
            self.w, self.h = 512, 512
        else:
            self.root = ['lytro']
            self.w, self.h = 552, 383
        self.num_view = 9
        self.AngualrViews = [i for i in range(self.num_view)]
        self.img_scale =  1 #   1 for small_baseline(default) <3.5px, 
                # 0.5 for large_baseline images   <  7px
        self.use_views = opt.use_views
        self.db = self._get_db() 
  
        # self.batch_size = opt.batch_size
        self._generate_512testdata()

    def __len__(self):
        return len(self.root)

    def __getitem__(self, index):
        d0 = self.view_0d[index, ...]
        d0 = torch.from_numpy(np.transpose(d0, (2, 0, 1)))
        d1 = self.view_90d[index, ...]
        d1 = torch.from_numpy(np.transpose(d1, (2, 0, 1)))
        d2 = self.view_45d[index, ...]
        d2 = torch.from_numpy(np.transpose(d2, (2, 0, 1)))
        d3 = self.view_m45d[index, ...]
        d3 = torch.from_numpy(np.transpose(d3, (2, 0, 1)))

        label = self.label[index, ...]
        label = torch.from_numpy(label).unsqueeze(0)
        return d0, d1, d2, d3, label
       

    def _get_db(self):
        db = []
        key = ['imgpath', 'data', 'disparity']
        item = {}
        tem_data = np.zeros((512, 512, 9, 9, 3),np.uint8)
        
        for subpath in self.root:
            # print(subpath)
            item['imgpath'] = subpath
            for i in range(self.num_view * self.num_view):
                try:
                    tmp  = np.float32(imageio.imread(subpath+'/input_Cam0%.2d.png' % i)) # load LF images(9x9) 
                except:
                    print(subpath+'/input_Cam0%.2d.png..does not exist' % i )
                tem_data[:,:,i//9,i-9*(i//9),:]=tmp  
                del tmp
            item['data'] = tem_data

            try:            
                tmp  = np.float32(read_pfm(subpath+'/gt_disp_lowres.pfm')) # load LF disparity map
            except:
                print(subpath+'/gt_disp_lowres.pfm..does not exist') 
            item['disparity'] = tmp
            del tmp   
            db.append(item) 
        return db

    def _generate_512testdata(self):
        input_size = self.w
        label_size = self.w
        num = len(self.db)
        
        # N*512*512*9 gray
        traindata_batch_90d = np.zeros((num,input_size,input_size,self.num_view),dtype=np.float32)
        traindata_batch_0d = np.zeros((num,input_size,input_size,self.num_view),dtype=np.float32)
        traindata_batch_45d = np.zeros((num,input_size,input_size,self.num_view),dtype=np.float32)
        traindata_batch_m45d = np.zeros((num,input_size,input_size,self.num_view),dtype=np.float32)
        
        # N*512*512 center view dis
        traindata_label_batchNxN = np.zeros((num,label_size,label_size))

        crop_half1 = int(0.5*(input_size-label_size))
        start1 = self.AngualrViews[0]
        end1 = self.AngualrViews[-1]
        for i in range(num):
            item = self.db[i]
            imgs = item['data']
            label = item['disparity']
            R, G, B = 0.299, 0.587, 0.114
            ix_rd = 0
            iy_rd = 0
            idx_start = 0
            idy_start = 0
            seq0_8 = np.array(self.AngualrViews) + ix_rd
            seq8_0 = np.array(self.AngualrViews[::-1]) + iy_rd
            traindata_batch_0d[i,:,:,:]=np.squeeze(R*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0_8,0].astype('float32')+
                                                 G*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0_8,1].astype('float32')+
                                                 B*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0_8,2].astype('float32'))
        
            traindata_batch_90d[i,:,:,:]=np.squeeze(R*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size, seq8_0, 4+iy_rd, 0].astype('float32')+
                                                 G*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size,seq8_0, 4+iy_rd, 1].astype('float32')+
                                                 B*imgs[idx_start: idx_start+input_size, idy_start: idy_start+input_size, seq8_0, 4+iy_rd, 2].astype('float32'))

            for k in range(start1, end1+1):
                traindata_batch_45d[i,:,:,k-start1]=np.squeeze(R*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, 8-k+ix_rd, k+iy_rd,0].astype('float32')+
                                                               G*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, 8-k+ix_rd, k+iy_rd,1].astype('float32')+
                                                               B*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, 8-k+ix_rd, k+iy_rd,2].astype('float32'))
                traindata_batch_m45d[i,:,:,k-start1] = np.squeeze(R*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, k+ix_rd, k+iy_rd,0].astype('float32')+
                                                               G*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, k+ix_rd, k+iy_rd,1].astype('float32')+
                                                               B*imgs[idx_start: idx_start+input_size,idy_start: idy_start+input_size, k+ix_rd, k+iy_rd,2].astype('float32'))
               
            if(num>=12 and label.shape[-1] == 9):                                
                traindata_label_batchNxN[i,:,:]=label[idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size, 4+ix_rd, 4+iy_rd]
            elif(self.num_view == 5):
                traindata_label_batchNxN[i,:,:]=label[idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size,0,0]
            else:
                traindata_label_batchNxN[i,:,:]=label[idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size]
        
        traindata_batch_90d=np.clip(np.float32((1/255)*traindata_batch_90d), 0., 1.)
        traindata_batch_0d =np.clip(np.float32((1/255)*traindata_batch_0d), 0., 1.)
        traindata_batch_45d=np.clip(np.float32((1/255)*traindata_batch_45d),  0., 1.)
        traindata_batch_m45d=np.clip(np.float32((1/255)*traindata_batch_m45d),  0., 1.)
        
        self.view_0d = traindata_batch_0d
        self.view_90d = traindata_batch_90d
        self.view_45d = traindata_batch_45d
        self.view_m45d = traindata_batch_m45d
        self.label = traindata_label_batchNxN
'''
    
def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data

def load_LFdata(dir_LFimages):    
       #traindata_all=np.zeros((len(dir_LFimages), 9, 9, 512, 512, 3),np.uint16)
       traindata_y=np.zeros((len(dir_LFimages),9,512,512,3),np.float32)
       traindata_x=np.zeros((len(dir_LFimages),9,512,512,3),np.float32)            
       traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)

       v_s=['004','013','022','031','040','049','058','067','076']
       h_s=['036','037','038','039','040','041','042','043','044']

       #batch_size=len(dir_LFimages)
       image_id=0

       for dir_LFimage in dir_LFimages:
          n=0
          for vs in v_s:
            traindata_y[image_id,n,:,:,:]=imread(dir_LFimage+'/input_Cam'+vs+'.png').astype('float32')
            n=n+1
          n=0
          for hs in h_s:
            traindata_x[image_id,n,:,:,:]=imread(dir_LFimage+'/input_Cam'+hs+'.png').astype('float32')
            n=n+1  
         
          traindata_label[image_id,:,:]=np.float32(read_pfm(dir_LFimage+'/gt_disp_lowres.pfm')).astype('float32')  
        
          image_id=image_id+1

       '''
       traindata_x=traindata_x.transpose(0,3,2,1,4)
       traindata_x=traindata_x.transpose(0,2,1,3,4)
       traindata_x=traindata_x.reshape(len(dir_LFimages),512,512*9,3)
         
        
       
       traindata_y=traindata_y.transpose(0,2,1,3,4)
       traindata_y=traindata_y.reshape(len(dir_LFimages),512*9,512,3)
       '''
       #traindata_label = traindata_label[:,:,:,np.newaxis]
        
       return traindata_x, traindata_y, traindata_label    #,len(dir_LFimages) ,traindata_x_new, traindata_y_new

def generate_traindata(traindata_x,traindata_y, traindata_label):  #, traindata_d_1,traindata_d_2

    mytransforms = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3), transforms.ToTensor()])
    
    input_size1=9
    input_size2=29
    input_size=29
    traindata_batch_x=np.zeros((input_size1,input_size2,3),dtype=np.float32)
    traindata_batch_y=np.zeros((input_size1,input_size2,3),dtype=np.float32)
    #traindata_batch_y_tmp=np.zeros((batch_size,input_size2,input_size1,3),dtype=np.float32)
    
    traindata_label_batchNxN=np.zeros((1,1,1),dtype=np.float32)
    

    '''
    
    #data augmentation refocusing
    cat = np.concatenate
    w = input_size1
    h = input_size1
    hw = int(w / 2)
    hh = int(h / 2)

    disp_range =[x for x in range(-3,4,1)]
    disp=np.random.choice(disp_range)
    
    for i in range(w):
      shift = disp * (i - hw)
      traindata_x[:, i, :, :, :] = cat([traindata_x[:, i, :, -shift:,:],traindata_x[:, i, :, :-shift,:]], -2)

    
    for i in range(h):
      shift = disp * (i - hh)
      traindata_y[:, i, :, :, :] = cat([traindata_y[:, i, -shift:, :, :],traindata_y[:, i, :-shift, :, :]], -3)

    # correct ground truth
        
    traindata_label -= float(disp)
    '''
    
    aa_arr =[x for x in range(0,16)]     

    image_id=np.random.choice(aa_arr)
    
    #image_id = ii
    
    aa_arr1 =[x for x in range(0,484)]
    id_start1 = np.random.choice(aa_arr1)
    aa_arr2 =[x for x in range(0,484)]
    id_start2 = np.random.choice(aa_arr2)
    
    
        
    traindata_batch_x = traindata_x[image_id:image_id+1, :,(id_start1+14):(id_start1+14)+1, (id_start2+14)-(input_size//2):(id_start2+14)+(input_size//2)+1,:].reshape(9,29,3)
                                            #G*traindata_x[image_id:image_id+1, id_start1+14-(input_size2//2):id_start1+14+(input_size2//2)+1, (id_start2+14)*9+4-(input_size1//2):(id_start2+14)*9+4+(input_size1//2)+1, 1].astype('float32')+
                                            #B*traindata_x[image_id:image_id+1, id_start1+14-(input_size2//2):id_start1+14+(input_size2//2)+1, (id_start2+14)*9+4-(input_size1//2):(id_start2+14)*9+4+(input_size1//2)+1, 2].astype('float32'))[:,:,np.newaxis]
    
    traindata_batch_y = traindata_y[image_id:image_id+1, :,(id_start1+14)-(input_size//2):(id_start1+14)+(input_size//2)+1, (id_start2+14):(id_start2+14)+1,:].reshape(9,29,3)
                                            #G*traindata_y[image_id:image_id+1, (id_start1+14)*9+4-(input_size1//2):(id_start1+14)*9+4+(input_size1//2)+1, id_start2+14-(input_size2//2):id_start2+14+(input_size2//2)+1, 1].astype('float32')+
    
    #traindata_batch_y[ii,:,:,:]= np.copy(np.rot90(traindata_batch_y_tmp[ii,:,:,:],1,(0,1)))

    traindata_batch_x = Image.fromarray(traindata_batch_x.astype(np.uint8))
    traindata_batch_y = Image.fromarray(traindata_batch_y.astype(np.uint8))
    
    traindata_label_batchNxN = traindata_label[image_id ,id_start1+14:id_start1+14+1,id_start2+14:id_start2+14+1].astype('float32')
        
    
    # traindata_batch_x=np.float32((1/255)*traindata_batch_x)
    # traindata_batch_y=np.float32((1/255)*traindata_batch_y)
    
    # traindata_batch_x=np.minimum(np.maximum(traindata_batch_x,0),1)
    # traindata_batch_y=np.minimum(np.maximum(traindata_batch_y,0),1)
    
    return mytransforms(traindata_batch_x) ,mytransforms(traindata_batch_y) , traindata_label_batchNxN#,traindata_batch_d_1,traindata_batch_d_2

def generate_valdata(valdata_x, valdata_y, valdata_label):            
    
    
    input_size=29
   
    num_x=482
    num_y=482
    num=0
    
    #valdata_batch_x=np.zeros((1*num_x*num_y,input_size1,input_size2,3),dtype=np.float32)
    #valdata_batch_y=np.zeros((1*num_x*num_y,input_size1,input_size2,3),dtype=np.float32)
    valdata_batch_x=np.zeros((1*num_x*num_y,9,input_size,3),dtype=np.float32)
    valdata_batch_y=np.zeros((1*num_x*num_y,9,input_size,3),dtype=np.float32)

    #valdata_batch_x_1=np.zeros((1*num_x*num_y,15,15,7),dtype=np.float32)
    #valdata_batch_y_1=np.zeros((1*num_x*num_y,15,15,7),dtype=np.float32)
    
    valdata_label_batch_482=np.zeros((num_x*num_y,1,1),dtype=np.float32)    

    
    
    #valdata_fill_y=valdata_fill_y[:,:,:,np.newaxis]

    
    for kk in range(0,num_x):
        for n in range(0,1):
            for jj in range(0,num_y,1):     #valdata_batch_y[num,:,:,:]=valdata_y[:, kk*9: kk*9+9, jj: jj+19, :].astype('float32')
                valdata_batch_x[num,:,:,:]=valdata_x[n,:, (kk+15) : (kk+15)+1, (jj+15)-(input_size//2):(jj+15)+(input_size//2)+1, :].reshape(9,29,3)
                valdata_batch_y[num,:,:,:]=valdata_y[n,:, (kk+15)-(input_size//2) : (kk+15)+(input_size//2)+1, (jj+15):(jj+15)+1, :].reshape(9,29,3)     
                valdata_label_batch_482[num,:,:]=valdata_label[n,kk+15:kk+15+1,jj+15:jj+15+1]
    
                num=num+1
      
     
          
    valdata_batch_x=np.float32((1/255)*valdata_batch_x)
    valdata_batch_y=np.float32((1/255)*valdata_batch_y)

    
    valdata_batch_x=np.minimum(np.maximum(valdata_batch_x,0),1)
    valdata_batch_y=np.minimum(np.maximum(valdata_batch_y,0),1)    
    
    
    
    return torch.FloatTensor(np.transpose(valdata_batch_x, (0,3,1,2))), \
           torch.FloatTensor(np.transpose(valdata_batch_y, (0,3,1,2))), \
           torch.FloatTensor(valdata_label_batch_482).unsqueeze(1)   #,valdata_batch_d_1,valdata_batch_d_2

def load_LF_valdata(dir_LFimages):   
    traindata_all_3=np.zeros((len(dir_LFimages), 9, 9, 512, 512, 3),np.uint16)
    traindata_all=np.zeros((len(dir_LFimages), 9, 9, 512, 512,3),np.uint16)
    traindata_y=np.zeros((len(dir_LFimages),512*9,512,3),np.uint16)
    traindata_x=np.zeros((len(dir_LFimages),512,512*9,3),np.uint16)
    
    traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)
    
    image_id=0
       
    for dir_LFimage in dir_LFimages:
          #print(dir_LFimage)
        
        traindata_all_3[image_id,:,:,:,:,:]=read_lightfield(dir_LFimage)
        traindata_all[image_id,:,:,:,:,:]=traindata_all_3[image_id,:,:,:,:,:]
    
        temp1=traindata_all[image_id,:,4,:,:,:].copy()
        temp1=temp1.transpose(2,1,0,3)
        temp1=temp1.transpose(1,0,2,3)
        traindata_x[image_id,:,:,:]=temp1.reshape(512,512*9,3)
    
        temp2=traindata_all[image_id,4,:,:,:,:].copy()
        temp2=temp2.transpose(1,0,2,3)
        traindata_y[image_id,:,:,:]=temp2.reshape(512*9,512,3)
        
        
        traindata_label[image_id,:,:]=np.float32(read_pfm(dir_LFimage+'/gt_disp_lowres.pfm'))  
        
        image_id=image_id+1
       #traindata_label = traindata_label[:,:,:,np.newaxis]
        
    return traindata_x, traindata_y, traindata_label, len(dir_LFimages)    #,traindata_x_new,traindata_y_new

class TrainDataset(Dataset):
    def __init__(self, opt, is_train=True, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = 32
        self.label_size = self.input_size

        print('Load training data...')
        dir_LFimages = [
            'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
            'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
            'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
            'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl']
        self.traindata_x, self.traindata_y, self.traindata_label = load_LFdata(dir_LFimages)
        # traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ = generate_traindata512(traindata_all,traindata_label, Setting02_AngualrViews) 
        print('Load training data... Complete')  


    def __len__(self):
        return 4000

    def __getitem__(self, index):
        trainx, trainy, trainlabel = generate_traindata(self.traindata_x, self.traindata_y, self.traindata_label)
        return trainx, trainy, trainlabel

class TestDataset(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])

        print('Load test data...') 
        self.dir_LFimages=[
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']

        self.valdata_x, self.valdata_y, self.valdata_label = load_LFdata(self.dir_LFimages)  # n*9*512*512*3 n*512*512
         # (valdata_90d, 0d, 45d, m45d) to validation or test      
        print('Load test data... Complete') 


    def __len__(self):
        return 1

    def __getitem__(self, index):
       generate_valdata(self.valdata_x, self.valdata_y, self.valdata_label)

