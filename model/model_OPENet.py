import torch.nn as nn
import torch
from torch.nn.functional import grid_sample,adaptive_avg_pool2d
from .loss import Lossv3,Lossv9,Lossv5,get_smooth_loss,noOPAL,Lossv4
from .context_adjustment_layer import *
from .basemodel import BaseModel, init_net, get_optimizer
from .layers import *
import time

IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]


class OPENetmodel(BaseModel): # Basemodel
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
  
        self.model_names = ['EPI']
        net = eval(self.opt.net_version)(opt, self.device,self.isTrain)
        # net = Unsup31_15_53(opt, self.device,self.isTrain)   # final syth:07-13-39  09-10-27(增强) real:08-1-12
        # net = Unsup27_16_16(opt, self.device,self.isTrain) # finetune syth:07-13-42
        # net = Unsup26_21_56(opt, self.device,self.isTrain) # fast syth:07-13-44  07-20-46  real: 08-01-23
        
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views

        self.center_index = self.use_v // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        self.lamda = opt.lamda

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = eval(self.opt.loss_version)(self.opt,self.device)
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            self.optimizers.append(self.optimizer)
        
    def set_input(self, inputs, epoch):
        self.epoch = epoch
        # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
        self.supervise_view = inputs[0].to(self.device)
        self.input = []
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, self.center_index,j])
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.center_index])    
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,j]) 
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j])     
        
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output, self.raw_warp_img = self.netEPI(self.input)  # G(A)
        self.test_loss_log = 0 

    def backward_G(self):
        # if self.epoch <= self.opt.n_epochs:
        #     self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        # else:
        #     self.loss_L1 = self.criterionL1_5(self.output, self.input[:9], self.input[9:18])
        # self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        self.loss_L1 = self.criterionL1(self.epoch,self.output, self.input[:9], self.input[9:18])
        if self.raw_warp_img is None:
             self.loss_total = self.loss_L1
        else:
            self.loss_raw = 0
            for views in self.raw_warp_img:
                self.loss_raw += self.criterionL1(self.epoch,views)  # 
            self.loss_total = 0.6*self.loss_L1 + 0.4*self.loss_raw
            # self.loss_total = self.loss_L1
        self.loss_smoothness = get_smooth_loss(self.output, self.center_input, self.lamda) 
        
        if 'smoothness' in self.loss_names and self.epoch > 2*self.opt.n_epochs:
            self.loss_total += self.loss_smoothness 
        self.loss_total.backward()
        
    def optimize_parameters(self):
        self.netEPI.train()
        self.forward()                   
        self.optimizer.zero_grad()       
        self.backward_G()                   
        self.optimizer.step()           


# finetune 1
# views 36
# cycle 1
# transformer 0
class Unsup27_16_16(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for _ in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, 64, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.finetune = ContextAdjustmentLayer()

    def forward(self, x):
        feats = []
        
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)
        # torch.cuda.synchronize()
        # end1 = time.time()
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], dim=2)  # B*8*9*h*w
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv)
        # torch.cuda.synchronize()
        # end3 = time.time()
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)
        # torch.cuda.synchronize()
        # end4 = time.time()

        occolusion = []
        disp = []
        mask = []
        raw_warp_img = []
        for j in range(self.n_angle):
            warpped_views = self.warp(disp_raw, x[self.use_v*j:self.use_v*(j+1)], IDX[j])
            if self.is_train:
                raw_warp_img.append(warpped_views) 
            occu = self.cal_occlusion(warpped_views)

            disp_final, occu_final = self.finetune(disp_raw, occu, x[4])
            # mask.append(torch.where(disp_final<0.03,1,0).float())
            disp.append(disp_final)
            # occolusion.append(occu_final)
        # mask = torch.where(torch.mean(torch.cat(mask, 1),1, True) < 1 ,0., 1.,) # all view==1, mask=1
        disp = torch.cat(disp, 1)
        disp_mean = torch.mean(disp, 1, True)   

        return disp_mean, raw_warp_img
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)

class Unsup27_16_16_nof(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for _ in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, 64, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.finetune = ContextAdjustmentLayer()

    def forward(self, x):
        feats = []
        
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)

        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], dim=2)  # B*8*9*h*w
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv)

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)


# finetune 1
# views 36
# cycle 1
# transformer 1
# final
class Unsup31_15_53(nn.Module):  
    def __init__(self,opt,device,is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.patch_size = opt.input_size
        self.center_index = self.use_v // 2
        # feature
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        # transformer
        ## v3 空间到特征
        self.transformer = nn.ModuleList()
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.transformer.append(transformer_layer(32,9,2,4)) # v2:36 v3:9
            self.block3d.append(nn.Sequential(nn.Conv3d(8, 64, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1),nn.LeakyReLU(0.2, True)))        
        # regression
        feats = 128
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        # finetune
        self.finetune = ContextAdjustmentLayer()
        
    def forward(self, x):
        feats = []
        # feature
        for xi in x:
            feat_i = self.feat_extract(xi)
            # atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            # feats.append(feat_i*atten_i)
            feats.append(feat_i)

        # transformer
        ## v3 10-31-15-53 分别trans24 分别fuse feats：8-64-32 空间到feats
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], 2)
            feats_tmp = rearrange(feats_tmp, 'b c a (h1 h) (w1 w) -> b (c h1 w1) a h w', h1=2, w1=2) # B*32*9*(h/2)*(w/2)
            feats_tmp = self.transformer[j](feats_tmp)
            feats_tmp = rearrange(feats_tmp, 'b (c h1 w1) a h w -> b c a (h1 h) (w1 w)', h1=2, w1=2) # B*8*36*(h/2)*(w/2)
            feats_tmp = self.block3d[j](feats_tmp) # B*32*36*(h/2)*(w/2)
            feats_tmp = torch.squeeze(feats_tmp, 2) 
            feats_angle.append(self.fuse3d[j](feats_tmp))

        # regression
        cv = torch.cat(feats_angle, 1) # b,128,h,w
        for j in range(4):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv )
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)
        # finetune
        occolusion = []
        disp = []
        mask = []
        raw_warp_img = []
        for j in range(self.n_angle):
            warpped_views = self.warp(disp_raw, x[self.use_v*j:self.use_v*(j+1)], IDX[j])
            if self.is_train:
                raw_warp_img.append(warpped_views) 
            occu = self.cal_occlusion(warpped_views)
            # self.vis(occu, j)

            disp_final, occu_final = self.finetune(disp_raw, occu, x[4])
            mask.append(torch.where(disp_final<0.03,1,0).float())
            disp.append(disp_final)
            # occolusion.append(occu_final)
        mask = torch.where(torch.mean(torch.cat(mask, 1),1, True) < 1 ,0., 1.,) # all view==1, mask=1
        disp = torch.cat(disp, 1)
        disp_mean = torch.mean(disp, 1, True)   
        return disp_mean, raw_warp_img
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)



# finetune 0
# views 18
# cycle 0
# transformer 0
class Unsup26_21_56(nn.Module):  # v3 轻量版 final
    def __init__(self,opt,device,is_train=True ):
        super(Unsup26_21_56, self).__init__() 
        self.is_train = is_train
        self.n_angle = 2
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        feats = 64
        # stack [b*8*9*h*w]*4
        # self.transformer = transformer_layer(32,self.use_v,2,4) 
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(2):
            '''
            self.block3d.append(nn.Sequential(nn.Conv3d(32, feats, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats//2, 1, 1),nn.LeakyReLU(0.2, True)))
            '''
            self.block3d.append(nn.Sequential(nn.Conv3d(8, feats, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats, 1, 1),nn.LeakyReLU(0.2, True)))
     
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        
        # finetune
        # self.warpping = Warpping('train')
        # self.finetune = ContextAdjustmentLayer()

        # warpping

        self.patch_size = opt.input_size

        self.center_index = self.use_v // 2
        
        # self.upsample = nn.Upsample(scale_factor=2)
        # if self.is_train:
        #     self.base_loss = torch.nn.L1Loss()
    
    def forward(self, x):

        B,C,H,W = x[0].shape

        feats = []
        feat = self.feat_extract(torch.cat(x[:18],0))

        feat = adaptive_avg_pool2d(feat, (1,1)) * feat
        for i in range(18):
            feats.append(feat[i*B:(i+1)*B,:,:,:])

        feats_angle = []
        for j in range(2):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], dim=2)  # B*8*9*h*w
            # feats_tmp = self.transformer(feats_tmp)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv )

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None
       
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

  
    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)

# finetune 0
# views 36
# cycle 1
# transformer 0
class Unsup22_13_01(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super(Unsup22_13_01, self).__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 32
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, 64, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 4
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.transformer = transformer_layer(32,self.use_v,2,4) 
        self.finetune = ContextAdjustmentLayer()
    def forward(self, x):
        feats = []

        for xi in x:
            feat_i = self.feat_extract(xi)
            atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            feats.append(feat_i * atten_i)

        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], dim=2)  # B*8*9*h*w
            # feats_tmp = self.transformer(feats_tmp)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))

        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv )

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

  
    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    


    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)



