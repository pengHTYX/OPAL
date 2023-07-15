from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset

import imageio
import tifffile as tif

def generate_hciold_for_train(traindata_all, traindata_label, input_size, ouc):
    """ initialize image_stack & label """
    traindata_batch = np.zeros(
        ( input_size, input_size,  9, 9,ouc),
        dtype=np.float32)
    traindata_batch_label = np.zeros((1, input_size, input_size))

    """ inital variable """
    crop_half1 = 0

    """ Generate image stacks"""
    sum_diff = 0
    while (sum_diff < 0.01 * input_size * input_size ):
        rand_3color = 0.05 + np.random.rand(3)
        rand_3color = rand_3color / np.sum(rand_3color)
        R = rand_3color[0]
        G = rand_3color[1]
        B = rand_3color[2]

        """
            We use totally 16 LF images,(0 to 15) 
            Since some images(4,6,15) have a reflection region, 
            We decrease frequency of occurrence for them. 
        """
        aa_arr = np.array([0, 1, 2])

        image_id = np.random.choice(aa_arr)
        ix_rd = 0
        iy_rd = 0

        kk = np.random.randint(17)
        scale = 1
        if (kk < 8):
            scale = 1
        elif (kk < 14):
            scale = 2
        elif (kk < 17):
            scale = 3
        
        hh, ww = np.shape(traindata_label[image_id])
        idx_start = np.random.randint(0, hh - scale * input_size)
        idy_start = np.random.randint(0, ww - scale * input_size)
 
        image_center = (1 / 255) * np.squeeze(
                R * traindata_all[image_id][idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 0].astype('float32') +
                G * traindata_all[image_id][idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 1].astype('float32') +
                B * traindata_all[image_id][idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 2].astype('float32'))

        sum_diff = np.sum(
                np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))
        
        if ouc == 1:
            traindata_batch= np.expand_dims(np.squeeze(
                R * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, :, 0].astype(
                    'float32') +
                G * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, :, 1].astype(
                    'float32') +
                B * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, :, 2].astype(
                    'float32')), -1) 
        else:
            traindata_batch =  traindata_all[image_id][idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, :, :].astype(
                    'float32')
               

        '''
            traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
        '''
        traindata_batch_label[0,:, :] = (1.0 / scale) * traindata_label[image_id][idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * input_size:scale,
                                                                idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * input_size:scale]

                                                                
    traindata_batch = np.float32((1 / 255) * traindata_batch)
    
    '''data argument'''
    # contrast
    gray_rand = 0.4 * np.random.rand() + 0.8
    traindata_batch= pow(traindata_batch, gray_rand)
    
    rotation_rand = np.random.randint(0, 5)
    # h*w*9*9*3   1*h*w
    if rotation_rand == 0:
        traindata_batch = np.transpose(traindata_batch, (4,0,1,2,3))
    elif rotation_rand == 1:# 90
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 1, (2, 3))) # w*h*9*9*3
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :]))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    elif rotation_rand == 2: # 180
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 2))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 2, (2, 3)))
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 2))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    elif rotation_rand == 3:# 270
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 3))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 3, (2, 3)))
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 3))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    else:# flip
        traindata_batch_tmp = np.copy(np.rot90(np.transpose(traindata_batch, (1, 0, 4, 2, 3))))
        traindata_batch = np.copy(np.transpose(traindata_batch_tmp[:,:,:,::-1],(2,0,1,3,4) ))# 3*w*h*9*9

        traindata_batch_label_tmp = np.copy(np.rot90(np.transpose(traindata_batch_label[0, :, :], (1,0))))
        traindata_batch_label[0,:,:] = traindata_batch_label_tmp# 1*w*h
     
    traindata_all = torch.FloatTensor(traindata_batch)
    label_all = torch.FloatTensor(traindata_batch_label)
    return traindata_all, label_all

def generate_traindata_for_train(traindata_all, traindata_label, input_size, label_size, batch_size,
                                 Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15, use_v, ouc):

 
    """ initialize image_stack & label """
    traindata_batch = np.zeros(
        ( input_size, input_size,  len(Setting02_AngualrViews), len(Setting02_AngualrViews),ouc),
        dtype=np.float32)
    traindata_batch_label = np.zeros((batch_size, label_size, label_size))

    """ inital variable """
    crop_half1 = int(0.5 * (input_size - label_size))

    """ Generate image stacks"""
    sum_diff = 0
    valid = 0

    while (sum_diff < 0.01 * input_size * input_size or valid < 1):
        rand_3color = 0.05 + np.random.rand(3)
        rand_3color = rand_3color / np.sum(rand_3color)
        R = rand_3color[0]
        G = rand_3color[1]
        B = rand_3color[2]

        """
            We use totally 16 LF images,(0 to 15) 
            Since some images(4,6,15) have a reflection region, 
            We decrease frequency of occurrence for them. 
        """
        aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                            0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                            0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        image_id = np.random.choice(aa_arr)

        if (len(Setting02_AngualrViews) == 9):
            ix_rd = 0
            iy_rd = 0

        kk = np.random.randint(17)
        scale = 1
        if (kk < 8):
            scale = 1
        elif (kk < 14):
            scale = 2
        elif (kk < 17):
            scale = 3

        idx_start = np.random.randint(0, 512 - scale * input_size)
        idy_start = np.random.randint(0, 512 - scale * input_size)
        valid = 1
        """
            boolmask: reflection masks for images(4,6,15)
        """
        if (image_id == 4 or 6 or 15):
            if (image_id == 4):
                a_tmp = boolmask_img4
                if (np.sum(a_tmp[
                            idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                            idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                    idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0
            if (image_id == 6):
                a_tmp = boolmask_img6
                if (np.sum(a_tmp[
                            idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                            idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                    idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0
            if (image_id == 15):
                a_tmp = boolmask_img15
                if (np.sum(a_tmp[
                            idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                            idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                    idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0

        if (valid > 0):
            
            image_center = (1 / 255) * np.squeeze(
                    R * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 0].astype('float32') +
                    G * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 1].astype('float32') +
                    B * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 2].astype('float32'))

            sum_diff = np.sum(
                    np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))
            
            if ouc == 1:
                traindata_batch= np.expand_dims(np.squeeze(
                    R * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 0].astype(
                        'float32') +
                    G * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 1].astype(
                        'float32') +
                    B * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 2].astype(
                        'float32')), -1) 
            else:
                traindata_batch = np.squeeze(
                    traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, :].astype(
                        'float32'))

            '''
                traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
            '''
            traindata_batch_label[0,:, :] = (1.0 / scale) * traindata_label[image_id,
                                                                    idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                    idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]

    traindata_batch = np.float32((1 / 255) * traindata_batch)
    
    '''data argument'''
    # contrast
    gray_rand = 0.4 * np.random.rand() + 0.8
    traindata_batch= pow(traindata_batch, gray_rand)
    
    rotation_rand = np.random.randint(0, 5)
    # h*w*9*9*3   1*h*w
    if rotation_rand == 0:
        traindata_batch = np.transpose(traindata_batch, (4,0,1,2,3))
    elif rotation_rand == 1:# 90
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 1, (2, 3))) # w*h*9*9*3
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :]))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    elif rotation_rand == 2: # 180
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 2))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 2, (2, 3)))
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 2))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    elif rotation_rand == 3:# 270
        traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 3))
        traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 3, (2, 3)))
        traindata_batch = np.transpose(traindata_batch_tmp5, (4,0,1,2,3))
        traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 3))
        traindata_batch_label[0, :, :] = traindata_label_tmp6
    else:# flip
        traindata_batch_tmp = np.copy(np.rot90(np.transpose(traindata_batch, (1, 0, 4, 2, 3))))
        traindata_batch = np.copy(np.transpose(traindata_batch_tmp[:,:,:,::-1],(2,0,1,3,4) ))# 3*w*h*9*9

        traindata_batch_label_tmp = np.copy(np.rot90(np.transpose(traindata_batch_label[0, :, :], (1,0))))
        traindata_batch_label[0,:,:] = traindata_batch_label_tmp# 1*w*h
     
    traindata_all = torch.FloatTensor(traindata_batch)
    label_all = torch.FloatTensor(traindata_batch_label)
    return traindata_all, label_all

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

def load_LFdata(root, dir_LFimages):    
    traindata_all=np.zeros((len(dir_LFimages), 512, 512, 9, 9, 3),np.uint8)
    traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)
    
    image_id=0
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread(root+dir_LFimage+'/input_Cam0%.2d.png' % i)) # load LF images(9x9) 
            except:
                print(root+dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
            traindata_all[image_id,:,:,i//9,i-9*(i//9),:]=tmp  
            del tmp
        try:            
            tmp  = np.float32(read_pfm(root+dir_LFimage+'/gt_disp_lowres.pfm')) # load LF disparity map
        except:
            print(root+dir_LFimage+'/gt_disp_lowres.pfm..does not exist' % i )            
        traindata_label[image_id,:,:]=tmp  
        del tmp
        image_id=image_id+1
    return traindata_all, traindata_label

def load_scanLF(root, dir_LFimages):    
    traindata_all = []
    traindata_label = []
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        try:
            tmp  = np.float32(imageio.imread(root+dir_LFimage+'/1.png')) # load LF images(9x9) 
        except:
            print(root+dir_LFimage+'/1.png does not exist...' )
        h,w = np.shape(tmp)
     
        del tmp
        if h%8 != 0:
            nn = h//8
            h = 8*nn
        if w%8 != 0:
            nn = w//8
            w = 8*nn
        traindata_=np.zeros( (h, w, 9, 9, 3),np.uint8)
        trainlabel_ = np.zeros((h, w),np.float32)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread(root+dir_LFimage+'/%d.png' % i)) # load LF images(9x9) 
            except:
                print(root+dir_LFimage+'/%d.png does not exist...' % (i+1))
            traindata_[:,:,i//9,i-9*(i//9),0]=tmp[:h,:w]
            traindata_[:,:,i//9,i-9*(i//9),1]=tmp[:h,:w]
            traindata_[:,:,i//9,i-9*(i//9),2]=tmp[:h,:w]
            del tmp
        traindata_all.append(traindata_)
        traindata_label.append(trainlabel_)
    return traindata_all, traindata_label

def load_hciold(root, dir_LFimages):    
    traindata_all = []
    traindata_label = []
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        try:
            tmp  = np.float32(imageio.imread(root+dir_LFimage+'/1.png')) # load LF images(9x9) 
        except:
            print(root+dir_LFimage+'/1.png does not exist...' )
        h,w,_ = np.shape(tmp)
     
        del tmp
        if h%8 != 0:
            nn = h//8
            h = 8*nn
        if w%8 != 0:
            nn = w//8
            w = 8*nn
        traindata_=np.zeros( (h, w, 9, 9, 3),np.uint8)
        trainlabel_ = np.zeros((h, w),np.float32)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread(root+dir_LFimage+'/%d.png' % (i+1))) # load LF images(9x9) 
            except:
                print(root+dir_LFimage+'/%d.png does not exist...' % (i+1))
            traindata_[:,:,i//9,i-9*(i//9),:]=tmp[:h,:w,:]
            del tmp
        try:            
            tmp = np.float32(tif.imread(root+dir_LFimage+'/disp.tif')) # load LF disparity map
        except:
            print(root+dir_LFimage+'/disp.tif..does not exist' )            
        trainlabel_ = tmp[:h,:w]
        del tmp
        traindata_all.append(traindata_)
        traindata_label.append(trainlabel_)
    return traindata_all, traindata_label

## hci_dataset
class TrainDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = self.opt.input_size + 2*opt.pad
        self.label_size = self.input_size
        self.use_v = opt.use_views
        self.inc = opt.input_c
        print('Load hci data...')
        dir_LFimages = [
            'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
            'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
            'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
            'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl']
        self.traindata_all, self.traindata_label = load_LFdata(opt.root, dir_LFimages)


        # traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ = generate_traindata512(traindata_all,traindata_label, Setting02_AngualrViews) 

        # load invalid regions from training data (ex. reflective region)   
        boolmask_img4= imageio.imread(opt.root + 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
        boolmask_img6= imageio.imread(opt.root + 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
        boolmask_img15=imageio.imread(opt.root + 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
        self.boolmask_img4  = 1.0*boolmask_img4[:,:,3]>0
        self.boolmask_img6  = 1.0*boolmask_img6[:,:,3]>0
        self.boolmask_img15 = 1.0*boolmask_img15[:,:,3]>0    

    def __len__(self):
        return 4000

    def __getitem__(self, index):
        return generate_traindata_for_train(self.traindata_all, self.traindata_label,
                                                                self.input_size,self.label_size,1,
                                                                self.Setting02_AngualrViews,
                                                                self.boolmask_img4,self.boolmask_img6,self.boolmask_img15, self.use_v, self.inc)    

# hci old fast
class TestDatase0(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hciold
        self.dir_LFimages=['/buddha', '/buddha2','/medieval',
            '/monasRoom', '/papillon',  '/stillLife'
            ] 
        self.valdata_all, self.valdata_label = load_hciold(self.dir_LFimages)
        
        if opt.input_c == 1 :
            self.valdata_all = np.expand_dims(self.valdata_all[:,:,:,:,:,0] * 0.299 + self.valdata_all[:,:,:,:,:,1] * 0.587 + self.valdata_all[:,:,:,:,:,2] * 0.114, -1)


    def __len__(self):
        return len(self.dir_LFimages)*4

    def __getitem__(self, index):
        """
        return:  0:input N*81*256*256  views=81
                 1:label N*256*256
                 2: name
        """
        ## hciold
        yu = index % 4
        index = index//4
        center = self.opt.use_views // 2
        name = self.dir_LFimages[index].split('/')  

        if yu==0:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:512, :512, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][:512,:512]) 
        elif yu==1:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][-512:, :512, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][-512:,:512]) 
        elif yu==2:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:512, -512:, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][:512,-512:])
        else:
            valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][-512:, -512:, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
            test_label =  torch.FloatTensor(self.valdata_label[index][-512:,-512:])

        return  valid_data,test_label, name[1]  # 81*512*512  512*512  array

# hci
class TestDataset1(Dataset):
    def __init__(self, opt, is_lytro=False, transform=None):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hci
        self.dir_LFimages=['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard',
            # 'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes'
            ] 
        self.valdata_all, self.valdata_label = load_LFdata(self.dir_LFimages)
       
        ## hciold
        # self.dir_LFimages=['/buddha', '/buddha2', '/monasRoom', '/papillon'] 
        
        # self.valdata_all, self.valdata_label = load_hciold(self.dir_LFimages)
        
        if opt.input_c == 1 :
            self.valdata_all = np.expand_dims(self.valdata_all[:,:,:,:,:,0] * 0.299 + self.valdata_all[:,:,:,:,:,1] * 0.587 + self.valdata_all[:,:,:,:,:,2] * 0.114, -1)


    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self, index):

        ## hciold
        # center = self.opt.use_views // 2
        # name = self.dir_LFimages[index].split('/')    
        # valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:, :, 4-center:5+center, 4-center:5+center,:]/255., (4,0,1,2,3)))
        # test_label =  torch.FloatTensor(self.valdata_label[index]) 
        # return  valid_data,test_label, name[1]  # 81*512*512  512*512  array

        ## hci
        center = self.opt.use_views // 2
        valid_data = self.valdata_all / 255.
        name = self.dir_LFimages[index].split('/')    
        valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.valdata_label[index, :,:]) 
        return  valid_data,test_label, name[1]  # 81*512*512  512*512  array

# scanLF
class TestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.input_size = opt.input_size
        
        print('Load test data...') 
        ## hci
        self.dir_LFimages=['test_HarryPotter_No0', 'test_THB_No0'] 
        self.valdata_all, self.valdata_label = load_scanLF(opt.root, self.dir_LFimages)
        if opt.input_c == 1 :
            self.valdata_all = np.expand_dims(self.valdata_all[:,:,:,:,:,0] * 0.299 + self.valdata_all[:,:,:,:,:,1] * 0.587 + self.valdata_all[:,:,:,:,:,2] * 0.114, -1)


    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self, index):
        center = self.opt.use_views // 2
        name = self.dir_LFimages[index]
        valid_data = torch.FloatTensor(np.transpose(self.valdata_all[index][:, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))
        test_label =  torch.FloatTensor(self.valdata_label[index]) 
        return  valid_data,test_label, name # 81*512*512  512*512  array