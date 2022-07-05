import importlib
from model.basemodel import BaseModel

from pathlib import Path
import logging
import time
import numpy as np
import torch

def create_model(opt):
    model_name = opt.model_name  # LFdepth
    model_filename = 'model.model_'+model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

def creat_logger(opt, mode='train'):
    root_output_dir = Path(opt.output_dir)  # output
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    model = opt.model_name  
    final_output_dir = root_output_dir / model
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True) # output/model
    
    if opt.time_str != '':
        time_str = opt.time_str
    else:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        tblog_dir = Path('tb') / model / time_str
        print('=> creating {}'.format(tblog_dir))
        tblog_dir.mkdir(parents=True, exist_ok=True)   # tb_log/model/time

    log_file = '{}_{}.log'.format(mode, time_str)
    final_log_file = final_output_dir / log_file  # output/model/train_time.log

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    

    return logger,  str(final_output_dir), str(final_log_file), str(tblog_dir), time_str


def tensor2im(input_image, mode):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # if is_I:
        #     image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
        # else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if mode == 'no':
            ma, mi = image_numpy.max(), image_numpy.min()
            image_numpy = (image_numpy-mi)/(ma-mi)
        image_numpy = image_numpy * 255.0  # post-processing: tranpose and scaling
        # elif label == 'real_BI'  or label == 'fake_BI':
        #     image_numpy = image_numpy * 255.0
        # elif label == 'real_BP' or label == 'fake_BP':
        #     image_numpy = image_numpy * np.pi
        #     ma, mi = image_numpy.max(), image_numpy.min()
        #     image_numpy = (image_numpy - mi)/(ma-mi) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy

def save_current_visual(visu, epoch, iter, writer, phase):
    '''save debug results to tensorboard'''
    modes = ['0,1', 'no', 'no']
    i = 0
    for name, tensor in visu.items():
        # img = tensor2im(name, tensor)
        save_name = phase + '_epoch_' + str(epoch) + '_' + str(iter) +'_'+name

        if len(tensor.shape) == 3:
            dataformats = 'HW'
        else:
            dataformats = 'CHW'
        if modes[i] == 'no': # convert to 0-1
            img_np = tensor[0].detach().cpu().float().numpy()
            ma, mi = img_np.max(), img_np.min()
            img_np = (img_np-mi)/(ma-mi)
            writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        else:
        # if name == 'center_input':
            writer.add_image(save_name, tensor[0], epoch)  # c*h*w [0,1]   dataformats : CHW, HWC, HW.
        '''https://pytorch.org/docs/1.7.1/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image'''
        i += 1 
        
        #         writer.add_image(save_name, tensor[0,j].unsqueeze(0), epoch)  # c*h*w [0,1]
            
import os
import cv2
import tifffile as tif
def save_img(opt, vis, img_name, time):
    '''save test result to ./results '''
    modes = ['0,1', 'no', 'no']
    save_dir = os.path.join(opt.results_dir, opt.model_name, opt.time_str)
    save_tifdir = os.path.join(opt.results_dir, opt.model_name, opt.time_str,'benchmark')
    if not os.path.exists(save_dir)  :
        os.makedirs(save_dir)
    if not os.path.exists(save_tifdir)  :
        os.makedirs(save_tifdir)   
    i = 0
    
    for vis_name, tensor in vis.items():
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        if i == 1:
            pred_numpy = tensor[0,0,:,:].cpu().float().numpy() 
            save_tif = os.path.join(save_tifdir,img_name + '_depth.tif')
            tif.imwrite(save_tif,pred_numpy)
        elif i==2:
            gt_numpy = tensor[0,0,:,:].cpu().float().numpy() 
        # save_name = img_name + '_'+vis_name+'.png'
        # save_img = tensor2im(tensor, modes[i])
        # save_fn = save_dir +'/'+ save_name
        # cv2.imwrite(save_fn,save_img[:,:,::-1])
        i += 1

    train_diff = np.abs(pred_numpy - gt_numpy)
    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    train_bp = (train_diff >= 0.07)
    bpr007 = 100 * np.average(train_bp)

    print('%s\tMSE*100 = %f\tbpr0.07 = %f' % (img_name, training_mean_squared_error_x100, bpr007))
   