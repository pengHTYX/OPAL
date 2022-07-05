from numpy import roots
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as RS
from skimage.metrics import structural_similarity as ssim
import lpips

from os import listdir
import torch
import glob
import os
from scipy.io import savemat

# psnr = 10 * log10(1/mes.item)

def cal_lpips(roots):
    # Initializing the model
    loss_fn = lpips.LPIPS(net='alex')

    if torch.cuda.is_available():
        loss_fn.cuda()

    subpath = ['checkpoints_0513_18_150', 'checkpoints_0513_12_150', 'checkpoints_0513']
    names = ['fake_BI.png', 'real_BI.png', 'fake_BP.png', 'real_BP.png']
  
    for k in range(len(roots)):
        subroot = roots[k]
        fakeI = sorted(glob.glob(os.path.join(subroot, '*'+names[0])))
        realI = [ele.replace(names[0], names[1]) for ele in fakeI]
        for j in range(len(fakeI)):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(fakeI[j]))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(realI[j]))

            if torch.cuda.is_available():
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            lips[j,k] = dist01.item()      
    savemat('lpips_0513_I.mat', {'data': lips})