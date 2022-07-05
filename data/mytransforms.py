import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import random
import numpy as np
from PIL import Image

def get_params(w, h, patch_size, scale, resize=True, crop=True, isrotate=True, isflip=True, grayscale=False):
    x = y = 0
    if crop:
        x = random.randint(0, np.maximum(0, w-patch_size))
        y = random.randint(0, np.maximum(0, h-patch_size))
   
    flip = False
    if isflip:
        flip = random.random() > 0.5
    
    rot = False
    if isrotate:
        rot = random.random() > 0.5

    return {'crop_pos': (x,y), 'crop_size':patch_size, 'scale':scale,
            'flip': flip, 'rot': rot, 'gray':grayscale, 'resize':resize}



def get_tranforms(params,  method=InterpolationMode.BICUBIC):
    transform_lr = []
    transform_hr = []
    transform_rgb = []

    mygray = transforms.Grayscale(1) 
    myresize = transforms.Resize(params['crop_size']//params['scale'], method) 
    myflip = transforms.Lambda(lambda img: __flip(img)) 
    myrot = transforms.Lambda(lambda img: __rotate(img)) 
    mycrop = transforms.Lambda(lambda img: __crop(img, params['crop_pos'], params['crop_size'])) 

    if params['gray']: 
        transform_rgb.append(mygray) 

    transform_lr.append(mycrop)
    transform_hr.append(mycrop)
    transform_rgb.append(mycrop)

    if params['resize']: 
        transform_lr.append(myresize) 
    
    if params['flip']:
        transform_lr.append(myflip)
        transform_hr.append(myflip)
        transform_rgb.append(myflip)
    
    if params['rot']:
        transform_lr.append(myrot)
        transform_hr.append(myrot)
        transform_rgb.append(myrot)

    transform_lr.append(transforms.ToTensor())
    transform_hr.append(transforms.ToTensor())
    transform_rgb.append(transforms.ToTensor())

    return transforms.Compose(transform_lr), transforms.Compose(transform_hr), transforms.Compose(transform_rgb)

def __rotate(img): # rotate 180
    return img.transpose(Image.ROTATE_180)

def __flip(img): # horizen flip
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def __crop(img, pos, size):
    if len(img.size) == 3:
        ow, oh = img.size[:2]
    else:
        ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        img = img.crop((x1, y1, x1 + tw, y1 + th))
    return img


