from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.parallel
import importlib

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from util import create_model, save_img, save_patch
from option import TestOptions
import os
import time

# cudnn related setting
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

def main():
    opt = TestOptions().parse() 
    data_lib = importlib.import_module('data.'+ opt.dataset_file)
    stanford_test = data_lib.TestStanford(opt) 
    valid_loader = torch.utils.data.DataLoader(stanford_test, batch_size=1, shuffle=False, num_workers=1)

    # print('The number of valid images = %d' % len(valid_dataset))
    save_dir = os.path.join(opt.results_dir, opt.model_name, opt.time_str)
    if not os.path.exists(save_dir)  :
        os.makedirs(save_dir) 

    model = create_model(opt)
    model.setup()
   
    # valid
    for data in valid_loader:
        info = data[-1]
        model.set_input(data[:-1], 0)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # save complete image
        _, _ = save_img(opt, visuals, info[0], save_dir)


    

       
        



if __name__ == '__main__':	

    main()


