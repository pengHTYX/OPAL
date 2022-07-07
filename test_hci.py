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
    hci_test = data_lib.ValDataset(opt) 
    valid_loader = torch.utils.data.DataLoader(hci_test, batch_size=1, shuffle=False, num_workers=1)

    # print('The number of valid images = %d' % len(valid_dataset))
    save_dir = os.path.join(opt.results_dir, opt.model_name, opt.time_str)
    if not os.path.exists(save_dir)  :
        os.makedirs(save_dir) 

    model = create_model(opt)
    model.setup()
   
    # valid
    mse_list, bpr_list, names = [], [], []
    for data in valid_loader:
        info = data[-1]
        model.set_input(data[:-1], 0)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        if isinstance(info, dict):
            # save patch image
            mse, bpr = save_patch(opt, visuals, info['name'][0], save_dir, info)
            names.append(info['name'][0])
        else:
            # save complete image
            mse, bpr = save_img(opt, visuals, info[0], save_dir)
            names.append(info[0])
        mse_list.append(mse)
        bpr_list.append(bpr)
    
    # log result
    with open(os.path.join('./hci_mse_bpr007.txt'), 'a') as f:
        f.write('{:-^100s}\n'.format(opt.time_str))
        _mse = 0
        _bpr = 0
        mse_str, bpr_str, name_str = '{:^10s}'.format('MSE'), '{:^10s}'.format('BPR'), '{:^10s}'.format('Scene')
        for i in range(len(mse_list)):
            _mse += mse_list[i]
            _bpr += bpr_list[i]
            name_str += '{:^10s}'.format(names[i])
            mse_str += '{:^10.2f}'.format(mse_list[i])
            bpr_str += '{:^10.2f}'.format(bpr_list[i])
        f.write(name_str + '{:^10s}\n'.format('mean')) 
        f.write(mse_str + '{:^10.2f}\n'.format(_mse/(i+1)))
        f.write(bpr_str + '{:^10.2f}\n'.format(_bpr/(i+1)))
        f.write('{:-^100s}\n'.format('-'))




    

       
        



if __name__ == '__main__':	

    main()


