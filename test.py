from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import savemat

import torch

import torch.nn.parallel
import importlib

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from util import create_model, save_img
from option import TestOptions



def main():
    opt = TestOptions().parse() 

    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    # logger, output_dir, log_dir, tb_dir = creat_logger(opt, 'test')
    
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    data_lib = importlib.import_module('data.'+ opt.dataset_file)
    
    valid_dataset = data_lib.TestDataset(opt)
   
    print('The number of valid images = %d' % len(valid_dataset))
  

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,

    )
    model = create_model(opt)
    model.setup()
    import time
    # valid
    # for j in range(50):
    for i,data in enumerate(valid_loader):
        if i >= opt.num_val:  # only apply our model to opt.num_test images.
            break
        name = data[-1][0]
        model.set_input(data[:-1], 0)  # unpack data from data loader
        model.test()           # run inference
        
        visuals = model.get_current_visuals()  # get image results
        save_img(opt, visuals, name, 0)

        # if i%4==0:
        #     save_img(opt, visuals, name+'_0', 0)
        # elif i%4==1:
        #     save_img(opt, visuals, name+'_1', 0)
        # elif i%4==2:
        #     save_img(opt, visuals, name+'_2', 0)
        # else :
        #     save_img(opt, visuals, name+'_3', 0)



if __name__ == '__main__':	

    main()


