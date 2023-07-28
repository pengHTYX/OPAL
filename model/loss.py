import torch
import torch.nn as nn
from torch.nn.functional import grid_sample,pad,softmax


IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]

class Lossv9(nn.Module):
    def __init__(self, opt,device):
        super(Lossv9, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2

        ones = torch.ones((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)
        zeros = torch.zeros((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)

        mask_1 = ones.repeat(1,9,1,1)
        mask_2 = torch.cat((zeros.repeat(1,1,1,1),ones.repeat(1,8,1,1)),1)
        mask_3 = torch.cat((zeros.repeat(1,2,1,1),ones.repeat(1,7,1,1)),1)
        mask_4 = torch.cat((zeros.repeat(1,3,1,1),ones.repeat(1,6,1,1)),1)
        mask_5 = torch.cat((zeros.repeat(1,4,1,1),ones.repeat(1,5,1,1)),1)
        mask_6 = torch.cat((ones.repeat(1,8,1,1),zeros.repeat(1,1,1,1)),1)
        mask_7 = torch.cat((ones.repeat(1,7,1,1),zeros.repeat(1,2,1,1)),1)
        mask_8 = torch.cat((ones.repeat(1,6,1,1),zeros.repeat(1,3,1,1)),1)
        mask_9 = torch.cat((ones.repeat(1,5,1,1),zeros.repeat(1,4,1,1)),1)

        # mask_up = torch.cat((ones.repeat(1,1,1,1),zeros.repeat(1,8,1,1)),1)
        # mask_down = torch.cat((zeros.repeat(1,8,1,1),ones.repeat(1,1,1,1)),1)

        self.mask = [mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8,mask_9]
        # self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9,mask_up,mask_down]

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        maps = []
        for x in x_list:
            map = torch.abs(x-x_list[4])
            map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
            maps.append(map)

        # maps B*9*H*W
        maps = torch.stack(maps,dim=1)
        distance_list = []

        for i in range(len(self.mask)):
            distance = self.mask[i]*maps
            distance = torch.sum(distance,dim=1)/torch.sum(self.mask[i],dim=1)
            distance_list.append(distance)

        diff = torch.abs(distance_list[8]-distance_list[4])
        diff = torch.where(diff<0.01,1,0)

        distances = torch.stack(distance_list,dim=1)
        distances, _ = torch.min(distances,dim=1)
        distances = distances*(1-diff)+distance_list[0]*diff
        loss = torch.mean(distances)

        return loss

class Lossv5(nn.Module):
    def __init__(self, opt,device):
        super(Lossv5, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2

        ones = torch.ones((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)
        zeros = torch.zeros((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)

        mask_1 = ones.repeat(1,9,1,1)
        # mask_2 = torch.cat((zeros.repeat(1,1,1,1),ones.repeat(1,8,1,1)),1)
        mask_3 = torch.cat((zeros.repeat(1,2,1,1),ones.repeat(1,7,1,1)),1)
        # mask_4 = torch.cat((zeros.repeat(1,3,1,1),ones.repeat(1,6,1,1)),1)
        mask_5 = torch.cat((zeros.repeat(1,4,1,1),ones.repeat(1,5,1,1)),1)
        # mask_6 = torch.cat((ones.repeat(1,8,1,1),zeros.repeat(1,1,1,1)),1)
        mask_7 = torch.cat((ones.repeat(1,7,1,1),zeros.repeat(1,2,1,1)),1)
        # mask_8 = torch.cat((ones.repeat(1,6,1,1),zeros.repeat(1,3,1,1)),1)
        mask_9 = torch.cat((ones.repeat(1,5,1,1),zeros.repeat(1,4,1,1)),1)


        # self.mask = [mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8,mask_9]
        self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9]
        # self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9,mask_up,mask_down]

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        maps = []
        for x in x_list:
            map = torch.abs(x-x_list[4])
            map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
            maps.append(map)

        # maps B*9*H*W
        maps = torch.stack(maps,dim=1)
        distance_list = []

        for i in range(len(self.mask)):
            distance = self.mask[i]*maps
            distance = torch.sum(distance,dim=1)/torch.sum(self.mask[i],dim=1)
            distance_list.append(distance)

        diff = torch.abs(distance_list[4]-distance_list[2])
        diff = torch.where(diff<0.01,1,0)

        distances = torch.stack(distance_list,dim=1)
        distances, _ = torch.min(distances,dim=1)
        distances = distances*(1-diff)+distance_list[0]*diff
        loss = torch.mean(distances)

        return loss

class Lossv3(nn.Module):
    def __init__(self, opt,de):
        super(Lossv3, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
  
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        map_up = torch.abs(x_list[0]-x_list[4])
        map_up = map_up[:,0,:,:]*0.299+map_up[:,1,:,:]*0.587+map_up[:,2,:,:]*0.114
        map_down = torch.abs(x_list[8]-x_list[4])
        map_down = map_down[:,0,:,:]*0.299+map_down[:,1,:,:]*0.587+map_down[:,2,:,:]*0.114

        mask_up = torch.unsqueeze(torch.where(map_up-map_down<self.alpha,1,0),dim=1)
        mask_down = torch.unsqueeze(torch.where(map_down-map_up<self.alpha,1,0),dim=1)

        loss = 0
        for j in range(4): ############## 5  7
            loss += self.base_loss(mask_up * x_list[j], mask_up * x_list[4]) ############ 2  3
        for j in range(5,9): ############## 5  7
            loss += self.base_loss(mask_down * x_list[j], mask_down * x_list[4]) ############ 2  3
        # loss = 0
        # for j in range(9): ############## 5   7
        #     loss += self.base_loss( x_list[j], x_list[4]) ############ 2  3
       
        return loss

class Lossv4(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, epoch, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp,epoch)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]),epoch)
        return total_loss
      
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list, epoch):
        maps = []
        cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114
        rate = 1/(cv_gray.detach()+2e-1)
        for i,x in enumerate(x_list):
            if i != 4:
                map = torch.abs(x-x_list[4])
                map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
                maps.append(map)
        maps = torch.stack(maps,dim=1)

        map_up = torch.mean(maps[:,0:4,:,:],dim=1)
        map_down = torch.mean(maps[:,4:,:,:],dim=1)
        if epoch>=30: map_circle = torch.mean(maps[:,2:6,:,:],dim=1)
        map_all = torch.mean(maps[:,:,:,:],dim=1)

        diff = torch.abs(map_up-map_down)
        diff = torch.where(diff<0.01,1,0)

        if epoch>=30: maps = torch.stack([map_up,map_down,map_circle,map_all],dim=1)
        else: maps = torch.stack([map_up,map_down,map_all],dim=1)

        map_min,_ = torch.min(maps,dim=1)
        loss = map_min*(1-diff)+map_all*diff
        loss = torch.mean(loss*rate)
       
        return loss

class noOPAL(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
  
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):

        loss = 0
        for j in range(9): ############## 5   7
            loss += self.base_loss( x_list[j], x_list[4]) ############ 2  3
       
        return loss




def get_smooth_loss(disp, img, lamda):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-lamda*grad_img_x)
    grad_disp_y *= torch.exp(-lamda*grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()