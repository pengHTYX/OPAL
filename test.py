from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    for i,data in enumerate(valid_loader):
        if i >= opt.num_val:  # only apply our model to opt.num_test images.
            break
        name = data[-1][0]
        model.set_input(data[:-1], 0)  # unpack data from data loader
        model.test()           # run inference
        
        visuals = model.get_current_visuals()  # get image results
        save_img(opt, visuals, name, 0)



if __name__ == '__main__':	

    main()


