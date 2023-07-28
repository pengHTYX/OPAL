from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy.io import savemat

import torch

import torch.nn.parallel
import importlib

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

from util import create_model, creat_logger
import torchvision.transforms as transforms
from option import TrainOptions
import os

def norm(tensor):
    return (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))

def main():
    opt = TrainOptions().parse() 
    opt.n_epochs_decay = int(3.5* opt.n_epochs)
    opt.lr_decay_iters = opt.n_epochs
    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    if not opt.debug: 
        logger, output_dir, log_dir, tb_dir, time_str = creat_logger(opt, 'train')
        opt.time_str = time_str
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    data_lib = importlib.import_module('data.'+ opt.dataset_file)
    train_dataset = data_lib.TrainDataset(opt, True)
            # normalize,])
    valid_dataset = data_lib.ValDataset(opt)
    print('The number of training images = %d' % len(train_dataset))
    print('The number of valid images = %d' % len(valid_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.is_shuffle,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    model = create_model(opt)
    model.setup()
    total_iters = 0  
    best_loss = 1e3  
    if not opt.debug:
        writer = writer_dict['writer']
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data, epoch)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_photometric', losses['L1'], global_steps)
                smoothloss_msg = 0
                if opt.losses.find('smooth') != -1:
                    writer.add_scalar('train_smoothness', losses['smoothness'], global_steps)
                    smoothloss_msg = losses['smoothness']
                writer_dict['train_global_steps'] = global_steps + 1
                
                msg = 'Epoch: [{0}/{6}][{1}/{2}]\t' \
                  'speeed: {3}s)\t' \
                  'Loss: {4}\t {5}'.format(epoch, i, len(train_loader), t_comp, losses['L1'], smoothloss_msg,opt.n_epochs + opt.n_epochs_decay)
                logger.info(msg)

        model.save_networks('latest')
        # valid
        # syth
        if epoch % 3 == 0:
            model.save_networks(str(epoch))
            mse_log = 0
            for i,data in enumerate(valid_loader):
            #     if i >= opt.num_val:  # only apply our model to opt.num_test images.
            #         break
                name = data[-1][0]
                model.set_input(data[:-1], epoch)
                model.test()
                visuals = model.get_current_visuals()  # get image results
                output = visuals['output'][0]
                # output = (output - torch.min(output))/(torch.max(output) - torch.min(output))
                # writer.add_image('v35*5_output_'+name, output, epoch)
                label = visuals['label'][0]
                diff = torch.abs(output-label)
                train_bp = torch.where(diff >= 0.07, 1, 0).float()
                mse_x100 = 100 * torch.mean(torch.square(diff))
                bad_pixel = 100 * torch.mean(train_bp)
                writer.add_scalar('v35*5_msex100_'+name, mse_x100, epoch)
                writer.add_scalar('v35*5_bad_pixel_'+name, bad_pixel, epoch)
                
                diff = (diff - torch.min(diff))/(torch.max(diff) - torch.min(diff))
                output = (output - torch.min(output))/(torch.max(output) - torch.min(output))
                writer.add_image('v35*5_output_'+name, output, epoch)
                writer.add_image('v35*5_diff_'+name, diff, epoch)
                
                mse_log += mse_x100.item()
       
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))



if __name__ == '__main__':	

    main()


