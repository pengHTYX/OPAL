
class Lossv2(nn.Module):
    def __init__(self, opt):
        super(Lossv2, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        
        total_loss = 0
        for  i in range(2):
            views = args[0][i*9:(i+1)*9]
            warp_list = self.warpping(args[1][0],views, IDX[i])
            if len(args) == 2:
                total_loss += self.cal_l1(warp_list[2:7])
            else:
                total_loss += self.cal_l1(warp_list[1:8], args[2][0])
                # total_loss += self.cal_l1(warp_list[2:7], args[2][1])
                total_loss += self.cal_l1(warp_list[3:6], args[2][1])
            
            # total_loss += self.cal_l1(self.warpping(args[2][0],views, IDX[i]),args[2][1])
            # total_loss += self.cal_l1(self.warpping(args[3][0],views, IDX[i]),args[3][1])
            # total_loss += self.cal_l1(self.warpping(args[4][0],views, IDX[i]),args[4][1])
        # total_loss = self.warpping(pred_disp, views)
        return total_loss
      
  
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,_,H,W = views_list[0].shape
       
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):                                                                                             #############
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  
        return tmp
    
    def cal_l1(self, x_list, mask=None):
        loss = 0
        NN = len(x_list)
        for j in range(NN):
            if mask is None:
                loss += self.base_loss(x_list[j], x_list[NN//2])
            else:
                loss += self.base_loss(torch.multiply(x_list[j], mask), torch.multiply(x_list[NN//2], mask))
            
                
        return loss/NN
 

class Feature_extract(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(Feature_extract, self).__init__()
        pad = kernel_size // 2 
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(conv_block(in_channels, 4, 3, 1, 1), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 8, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
        self.layer4 = self._make_layer(16, 16, 2, 1, 1)      ######  源代码为dilation=2

        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=8))
        self.branch4 = nn.Sequential(nn.AvgPool2d(16,16), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=16))

        self.convx = nn.Conv2d(40,64, 1,1)
        self.convy = nn.Conv2d(40,64, 1,1)
        self.lastconv = nn.Sequential(conv_2d(64, 16, 3, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(16,4,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        x = torch.cat([l2, l4, self.branch1(l4), self.branch2(l4), self.branch3(l4), self.branch4(l4)], 1)
        x = self.lastconv(x)
        return x


class basic(nn.Module):
    def __init__(self,in_channels=64, feats=64):
        super(basic, self).__init__()
        
        self.res1 = nn.Sequential(conv_3d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_3d(feats, feats, 3, 1))
        self.relu1 = nn.LeakyReLU(0.2, True)
        # self.res2 = nn.Sequential(conv_3d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
        #                           conv_3d(feats, feats, 3, 1))
        # self.relu2 = nn.LeakyReLU(0.2, True)                         
        self.last = nn.Sequential(conv_3d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  nn.Conv3d(feats, 1, 3, 1, 1))
        

    def forward(self, x): 
        x = self.relu1(self.res1(x)+x)
        # x = self.relu2(self.res2(x)+x)
        x = self.last(x)
        return x

class basic2d(nn.Module):
    def __init__(self,in_channels=64, feats=64):
        super(basic2d, self).__init__()
        
        self.res1 = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_2d(feats, feats, 3, 1))
        self.relu1 = nn.LeakyReLU(0.2, True)
        
        self.res2 = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_2d(feats, feats, 3, 1))
        self.relu2 = nn.LeakyReLU(0.2, True)    

        self.res3 = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  conv_2d(feats, feats, 3, 1))
        self.relu3 = nn.LeakyReLU(0.2, True)    

        self.last = nn.Sequential(conv_2d(feats, feats, 3, 1),nn.LeakyReLU(0.2, True), 
                                  nn.Conv2d(feats, 2, 3, 1, 1))
        
    def forward(self, x): 
        x = self.relu1(self.res1(x)+x)
        x = self.relu2(self.res2(x)+x)
        x = self.relu3(self.res3(x)+x)
        x = self.last(x)
        return x

class CostRegNet(nn.Module):
    def __init__(self, inc=32, ouc=1):
        super(CostRegNet, self).__init__()
        feats = 32
        self.conv0 = ConvBnReLU3D(inc, feats)

        self.conv1 = ConvBnReLU3D(feats, feats*2, stride=(1, 2, 2))
        self.conv2 = ConvBnReLU3D(feats*2, feats*2)

        self.conv3 = ConvBnReLU3D(feats*2, feats*4, stride=(1, 2, 2))
        self.conv4 = ConvBnReLU3D(feats*4, feats*4)

        self.conv5 = ConvBnReLU3D(feats*4, feats*8, stride=(1, 2, 2))
        self.conv6 = ConvBnReLU3D(feats*8, feats*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(feats*8, feats*4, kernel_size=3, padding=1, output_padding=(0, 1,1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(feats*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(feats*4, feats*2, kernel_size=3, padding=1, output_padding=(0, 1,1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(feats*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(feats*2, feats, kernel_size=3, padding=1, output_padding=(0, 1,1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(feats),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(feats, ouc, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        #print(conv4.shape)
        #print(self.conv7(x).shape)

        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)

        del conv2
        del conv4
        torch.cuda.memory_allocated()
        torch.cuda.memory_reserved() # torch.cuda.memory_cached()
        torch.cuda.empty_cache()

        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class CostRegNet2(nn.Module):
    def __init__(self, inc=32, ouc=1, feats=32):
        super(CostRegNet2, self).__init__()
        
        self.conv0 = ConvBnReLU(inc, feats)

        self.conv1 = ConvBnReLU(feats, feats*2, stride=2)
        self.conv2 = ConvBnReLU(feats*2, feats*2)

        self.conv3 = ConvBnReLU(feats*2, feats*4,stride=2)
        self.conv4 = ConvBnReLU(feats*4, feats*4)

        # self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        # self.conv6 = ConvBnReLU3D(64, 64)

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(feats*4, feats*2, kernel_size=2,  padding=0,  stride=2, bias=False),
            nn.BatchNorm2d(feats*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(feats*2, feats, kernel_size=2, padding=0,  stride=2, bias=False),
            nn.BatchNorm2d(feats),
            nn.ReLU(inplace=True))
        if ouc == 1:
            self.prob = nn.Conv2d(feats, ouc, 3, stride=1, padding=1)
        else:
            self.prob = nn.Sequential(nn.Conv2d(feats, ouc, 3, stride=1, padding=1), nn.Softmax(1))
        
    def forward(self, x):
        conv0 = self.conv0(x)
      
        conv2 = self.conv2(self.conv1(conv0))
        x = self.conv4(self.conv3(conv2))
        # x = self.conv6(self.conv5(conv4))
        #print(conv4.shape)
        #print(self.conv7(x).shape)

        # x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)

        del conv2
        torch.cuda.memory_allocated()
        torch.cuda.memory_reserved()#  torch.cuda.memory_cached()
        torch.cuda.empty_cache()

        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class EPINet(nn.Module):
    def __init__(self, n_views, ngf, n_stream, num_block):
        super(EPINet, self).__init__() 
        
        self.feat_extract = Feature_extract(in_channels=1)
        
        self.relu = nn.LeakyReLU(0.2, True)
        self.fc1 = nn.Conv3d(4*81, 170, 1,1)
        self.fc2 = nn.Conv3d(170, 15, 1, 1)
        
        self.basic = basic()
        self.softmax = nn.Softmax(dim=2)

    
    def forward(self, x):
        feats = []
        for xi in x:
            feats.append(self.feat_extract(xi))
        cv = self._cost_volumn_(feats)  # N*324*D*H*W

        # cv: N*324*D*H*W  
        # attention : N*324*1*1*1 
        cv, attention = self.channel_attention(cv)
        
        cost = self.basic(cv) #  N*1*D*H*W  
        cost = self.softmax(cost).squeeze(1) # N*D*H*W
       
        pred = self.disparitygression(cost) # N*H*W
        
        # return pred + x[40].squeeze(1)
        return pred 

    def channel_attention(self, cost_volumn):
        # N*C*D*H*W
        x = adaptive_avg_pool3d(cost_volumn, (1,1,1)) # N*C*1*1*1
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) # N*15*1*1*1
        
        x = torch.cat([x[:, 0:5,:, :, :], x[:, 1:2,:, :, :], x[:, 5:9,:, :, : ], x[:, 2:3,:, :, : ],
                        x[:,  6:7,:, :, :,], x[:, 9:12, :, :, :], x[:, 3:4, :, :, :], x[:, 7:8, :, :, :],
                        x[:, 10:11,:, :, : ], x[:, 12:14, :, :, :], x[:, 4:5, :, :, :], x[:, 8:9, :, :, :],
                      x[:, 11:12, :, :, :], x[:, 13:15, :, :, :]], 1)
        x = torch.reshape(x, (x.shape[0],1, 5, 5))
        x = pad(x, (0,4,0,4), 'reflect')
        attention = torch.reshape(x, (x.shape[0],81,1,1,1))
        x = attention.repeat(1,4,1,1,1)
        return torch.multiply(x, cost_volumn), attention

    def _cost_volumn_(self, inputs):
        disparity_costs = []
        input_shape = inputs[0].shape
        for d in range(-4, 5):
            if d == 0:
                tmp_list = []
                for i in range(len(inputs)):
                    tmp_list.append(inputs[i])
            else:
                tmp_list = []
                for i in range(len(inputs)):
                    (v, u) = divmod(i, 9)
                    theta = torch.tensor([[[1,0,d * (u - 4)],    # 右负
                                          [0,1,d * (v - 4)]]], dtype=torch.float).to(inputs[0])   # 下负
                    grid =  affine_grid(torch.cat([theta] * input_shape[0], 0), input_shape, align_corners=True)
                    tensor = grid_sample(inputs[i], grid,align_corners=True)
                    # tensor = tf.contrib.image.translate(inputs[i], [d * (u - 4), d * (v - 4)], 'BILINEAR')
                    tmp_list.append(tensor)
            cost = torch.cat(tmp_list, 1) # N*324*H*W
            disparity_costs.append(cost)
        cost_volumn = torch.stack(disparity_costs, 2) # N*324*D*H*W
        return cost_volumn
    
    def disparitygression(self, input):
        N, D, H, W = input.shape
        disparity_values = torch.linspace(-4,4,9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = torch.tile(x, (N, 1, H, W))
        out = torch.sum(torch.multiply(input, x), 1)
        return out 


class UnsuperviseNetv4(nn.Module):  # v4
    def __init__(self,opt,is_train=True):
        super(UnsuperviseNetv4, self).__init__() 
        self.is_train = is_train
        self.opt = opt
        self.n_angle = 4
        self.use_v = opt.use_views
        self.center_index = self.use_v // 2
        self.feat_extract = Feature(in_channels= opt.input_c, out_channels=8)
        
        
        # 得到attention1
        self.block3d = nn.Sequential(nn.Conv3d(8, self.use_v, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True))
        self.fuse3d = nn.Sequential(nn.Conv3d(self.use_v, self.use_v, 1, 1),nn.LeakyReLU(0.2, True))
        
        feats = 8*self.use_v # 8*9
        self.fuse2d = nn.ModuleList()
        # 得到attention2
        for j in range(3):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1),nn.LeakyReLU(0.2, True) ))

        self.fuse3 = nn.Conv2d(feats, self.use_v, 3, 1, padding=1)
        self.relu3 = nn.Sigmoid()
        
        # unet预测
        self.unet = CostRegNet2(inc=feats, feats=64, ouc=opt.out_c)
        
        # finetune
        self.finetune = ContextAdjustmentLayerv2()
      
    
    def forward(self, x):

        feats = []
       
        for xi in x:
            feats.append(self.feat_extract(xi)) # B*32*H*W
        
        # self.cal_feat_grad(feats)
     
        feats_tmp = torch.stack(feats, dim=2)  # B*8*9*h*w
        feats_tmp = self.block3d(feats_tmp)  # B*9*1*h*w
        atten1 = self.fuse3d(feats_tmp)  # 
        for i,feat in enumerate(feats):
            feats[i] = torch.multiply(feat, atten1[:,i,:,:,:])
        
        feats_new = torch.cat(feats, 1)
        for j in range(3):
            feats_new = self.fuse2d[j](feats_new)
        atten2 = self.relu3(self.fuse3(feats_new)) # B*9*H*W  [0,1]
        for i,feat in enumerate(feats):
            feats[i] = torch.multiply(feat, atten2[:,i,:,:].unsqueeze(1))
        feats_new = torch.cat(feats, 1)

        disp = self.unet(feats_new)
        if self.opt.out_c != 1:
            disp = self.disparitygression(disp)
        disp_final = self.finetune(disp, x[4])
        
        # return pred + x[40].squeeze(1)
        return disp_final, atten2
    
    def disparitygression(self, input):
        N, D, H, W = input.shape
        disparity_values = torch.linspace(-4,4,9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = torch.tile(x, (N, 1, H, W))
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    
    def vis(self,grad,j):
        import numpy as np
        import cv2
        grad = grad[0,0].cpu().float().numpy()
        ma,mi = np.max(grad), np.min(grad)
        cv2.imwrite('results/%d.png'% j, (grad-mi)/(ma-mi)*255)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:self.use_v-1]), dim=[1,4]) # B*H*W
        return grad.unsqueeze(1)

class UnsuperviseNetoldv2(nn.Module): # v2
    def __init__(self, opt, is_train):
        super(UnsuperviseNetoldv2, self).__init__() 
        
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        self.use_v = opt.use_views
        
        feats = 32
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, feats, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 4
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        
        
        # unet分支
        # self.unet3d = nn.ModuleList()
        # for j in range(4):
        #     self.unet3d.append(CostRegNet())
        self.unet2d = CostRegNet2(inc=feats, ouc=1)

    def cal_feat_var(self, feats):
        import numpy as np
        import cv2
        B,C,H,W = feats[0].shape
        self.feats_var = []
        for  j in range(4):
            tmp = torch.cat(feats[j*9:(j+1)*9], 1)
            mean = torch.mean(tmp, 1, keepdim=True)
            var = torch.sqrt(torch.mean(torch.square(tmp-mean),1))[0]
            var = var.cpu().float().numpy()
            ma,mi = np.max(var), np.min(var)
            cv2.imwrite('results/%d.png'% j, (var-mi)/(ma-mi)*255)

    def forward(self, x):
        B,C,H,W = x[0].shape
       
        # 特征提取
        feats = []
        for xi in x:
            feat_i = self.feat_extract(xi)
            atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            feats.append(feat_i * atten_i)
       
        # unet branch
        # feats3d = []
        # for j in range(4):
        #     feats_tmp = torch.stack(feats[j*9:(j+1)*9], dim=2)  # B*8*9*h*w
        #     feats_tmp = self.unet3d[j](feats_tmp).squeeze(1)  # B*9*h*w
        #     feats3d.append(feats_tmp)
        # cv1 = torch.cat(feats3d, 1)  # B*36*h*w
        # disp1 = self.unet2d(cv1)

        # 回归分支
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v*j:self.use_v*(j+1)], dim=2)  # B*8*9*h*w
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) 
            feats_angle.append(self.fuse3d[j](feats_tmp))# B*32*h*w

        cv2 = torch.cat(feats_angle, 1) # B*128*h*w
        disp1 = self.unet2d(cv2)
        for j in range(3):
            cv2 = self.fuse2d[j*2+1](self.fuse2d[j*2](cv2) + cv2 )
       
        # disp = self.fuse3(cv2)
        prob = self.relu3(self.fuse3(cv2)) # B*9*H*W
        disp2 = self.disparitygression(prob)
        coresponding = torch.max(prob, 1, True)
    
        disp = coresponding[0] * disp1 + (1-coresponding[0])*disp2
        return disp            

    def disparitygression(self, input):
        N, D, H, W = input.shape
        disparity_values = torch.linspace(-4,4,9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = torch.tile(x, (N, 1, H, W))
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

class UnsuperviseNetv2(nn.Module): # new v2
    def __init__(self, opt, is_train):
        super(UnsuperviseNetv2, self).__init__() 
        
        self.feat_extract = Feature(in_channels=3,out_channels=8)
        self.use_v = 9 # opt.use_views
        
        feats = 64
        # self.block3d = nn.ModuleList()
        # for k in [36,28,20,12]:
        #     self.block3d.append(nn.Sequential(nn.Conv2d(k*4,feats,3,1,1),nn.BatchNorm2d(feats), nn.LeakyReLU(0.2, True), 
        #                                       nn.Conv2d(feats, feats, 3,1,1),nn.BatchNorm2d(feats), nn.LeakyReLU(0.2, True),
        #                                       nn.Conv2d(feats, feats, 3,1,1),nn.BatchNorm2d(feats), nn.LeakyReLU(0.2, True)))

        # self.basic = basic2d()
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

        self.cost_regular = CostRegNet(32, 1)
        self.pool3d = nn.AvgPool3d((3, 1, 1), stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        B,C,H,W = x[0].shape
        # 特征提取
        feats = []
        for xi in x:
            feat_i = self.feat_extract(xi)
            # atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            feats.append(feat_i)
        
        cv9 = torch.stack([torch.cat(feats[j:36:9], 1) for j in range(9)], 2)    # B*32*9*H*W
        # cv9 = self.block3d[0](cv9) # 
        cv_reg = self.cost_regular(cv9).squeeze(1)
        cv_reg = self.softmax(cv_reg)
        disparity_values = torch.linspace(-4,4,9).to(x[0])
        disp = self.disparitygression(cv_reg,disparity_values)
        mask1 = torch.where(torch.abs(disp)>=1,1,0).float()
        mask3 = torch.where(torch.abs(disp)<1, 1, 0).float()
        # mask2 = torch.ones((B,1,H,W)).to(x[0]) - mask1-mask3

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 3 * self.pool3d(pad(cv_reg.unsqueeze(1), pad=(0, 0, 0, 0, 1, 1))).squeeze(1)
            depth_index = self.disparitygression(cv_reg, value=torch.arange(9, device=x[0].device, dtype=torch.float)).long()
          
            depth_index = depth_index.clamp(min=0, max=8)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index)
        
        
        # basef = 4

        # cv9 = torch.cat(feats, 1)    # 144
        # cv9 = self.block3d[0](cv9)
        # cv9 = self.basic(cv9)
        # disp, photometric_confidence = cv9[:,0,:,:].unsqueeze(1), self.sigmoid(cv9[:,1,:,:].unsqueeze(1))
        ''' 
        start, end =  1, 8      
             
        cv7 =    torch.cat([cv9[:,start*basef:end*basef,:,:,:], \
                            cv9[:,(start+9)*basef:(end+9)*basef,:,:,:], \
                            cv9[:,(start+18)*basef:(end+18)*basef,:,:,:], \
                            cv9[:,(start+27)*basef:(end+27)*basef,:,:,:]], 1)  #B*112*9*H*W
        
        start, end =  2, 7
        cv5 =    torch.cat([cv9[:,start*basef:end*basef,:,:,:], \
                            cv9[:,(start+9)*basef:(end+9)*basef,:,:,:], \
                            cv9[:,(start+18)*basef:(end+18)*basef,:,:,:], \
                            cv9[:,(start+27)*basef:(end+27)*basef,:,:,:]], 1)  #B*80*9*H*W
        start, end =  3, 6                 
        cv3 =    torch.cat([cv9[:,start*basef:end*basef,:,:,:], \
                            cv9[:,(start+9)*basef:(end+9)*basef,:,:,:], \
                            cv9[:,(start+18)*basef:(end+18)*basef,:,:,:], \
                            cv9[:,(start+27)*basef:(end+27)*basef,:,:,:]], 1)  #B*48*9*H*W
        cv9 = self.basic(self.trans1(cv9)).squeeze(1)
        cv7 = self.basic(self.trans2(cv7)).squeeze(1)
        cv5 = self.basic(self.trans3(cv5)).squeeze(1)
        cv3 = self.basic(self.trans4(cv3)).squeeze(1)
        
        cv9 = torch.cat(feats, 1)    # 144
        cv7 = torch.cat(feats[1:8] + feats[10:17] + feats[19:26] + feats[28:35], 1 ) # 112
        cv5 = torch.cat(feats[2:7] + feats[11:16] + feats[20:25] + feats[29:34], 1 ) # 80
        cv3 = torch.cat(feats[3:6] + feats[12:15] + feats[21:24] + feats[30:33], 1 ) # 48

        cv9 = self.block3d[0](cv9)
        cv7 = self.block3d[1](cv7)
        cv5 = self.block3d[2](cv5)
        cv3 = self.block3d[3](cv3)
        

    
        disp_k = []
        conf_k = []
        for cv in [cv9, cv7, cv5, cv3]:
            cv = self.basic(cv)
            disp_k.append(cv[:,0,:,:])
            conf_k.append(self.sigmoid(cv[:,1,:,:]))

        p1 = self.softmax(torch.stack(conf_k, 1))
        p = self.softmax(torch.multiply(torch.where(p1<0.25, 0., 1.), p1))
        disp = torch.stack(disp_k, 1)  # B*4*H*W
        disp = torch.sum(torch.multiply(disp, p), 1, True)
        return disp, (disp_k[0].unsqueeze(1), p1[:,0,:,:].unsqueeze(1)), \
                     (disp_k[1].unsqueeze(1), p1[:,1,:,:].unsqueeze(1)), \
                     (disp_k[2].unsqueeze(1), p1[:,2,:,:].unsqueeze(1)), \
                     (disp_k[3].unsqueeze(1), p1[:,3,:,:].unsqueeze(1))      
        '''
        return (disp, photometric_confidence), (mask1, mask3)
    def _varience_cost_volumn_(self, inputs):# [B*8*H*W]*36
        view_n = len(inputs)
        B,C,H,W = inputs[0].shape
        disp_sum = torch.zeros((B,C,9,H,W)).to(inputs[0])
        disp_varience = torch.zeros((B,C,9,H,W)).to(inputs[0])
        disp_list = []
        for i in range(view_n):
            disparity_costs = []
            uu,vv = divmod(i, 9)
            v, u = divmod(IDX[uu][vv], 9)
            for d in range(-4, 5):
                theta = torch.tensor([[[1,0,d * (u - 4)],    # 右负
                                          [0,1,d * (v - 4)]]], dtype=torch.float).to(inputs[0])   # 下负
                grid =  affine_grid(torch.cat([theta] * B, 0), (B,C,H,W), align_corners=True)
                tensor = grid_sample(inputs[i], grid,align_corners=True)
                disparity_costs.append(tensor)
            disparity_costs = torch.stack(disparity_costs, 2) # B*8*9*H*W
            disp_list.append(disparity_costs)
            disp_sum += disparity_costs
        disp_sum = disp_sum.div_(view_n)

        for volumn in disp_list:
            disp_varience = disp_varience + (volumn-disp_sum).pow_(2)
        return disp_varience.div_(view_n) 

    def _cost_volumn_(self, inputs): # [B*4*H*W]*36
        disparity_costs = []
        input_shape = inputs[0].shape
        for d in range(-4, 5):
            if d == 0:
                tmp_list = []
                for i in range(len(inputs)):
                    tmp_list.append(inputs[i])
            else:
                tmp_list = []
                for i in range(len(inputs)):
                    v, u = divmod(i, 9)
                    theta = torch.tensor([[[1,0,d * (u - 4)],    # 右负
                                          [0,1,d * (v - 4)]]], dtype=torch.float).to(inputs[0])   # 下负
                    grid =  affine_grid(torch.cat([theta] * input_shape[0], 0), input_shape, align_corners=True)
                    tensor = grid_sample(inputs[i], grid,align_corners=True)
                    # tensor = tf.contrib.image.translate(inputs[i], [d * (u - 4), d * (v - 4)], 'BILINEAR')
                    tmp_list.append(tensor)
            cost = torch.cat(tmp_list, 1) # N*144*H*W
            disparity_costs.append(cost)
        cost_volumn = torch.stack(disparity_costs, 2) # N*144*D*H*W
        return cost_volumn

    def disparitygression(self, input,value):
        N, D, H, W = input.shape
        
        x = value.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = torch.tile(x, (N, 1, H, W))
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

# a = [torch.randn((4,3,64,64)) for _ in range(36)]
# mymodel = UnsuperviseNetv2('train', True)
# b = mymodel(a)


class UnsuperviseNetv1(nn.Module):  # v1
    def __init__(self, is_train):
        super(UnsuperviseNetv1, self).__init__() 
        feats = 64
        self.feat_extract = Feature(in_channels=3, out_channels=8)
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, feats, (9,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 4
        self.fuse2d = nn.ModuleList()
        for j in range(3):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)

    def cal_feat_grad(self, feats):
        import numpy as np
        import cv2
      
        B,C,H,W = feats[0].shape
        self.feats_var = []
     
        for  j in range(4):
            tmp = torch.cat(feats[j*9:(j+1)*9], 1) # B*9*H*W
            grad = torch.mean(torch.abs(tmp[:,1:,:,:]-tmp[:,:9*C-1,:,:]),1)
            grad = grad[0].cpu().float().numpy()
            ma,mi = np.max(grad), np.min(grad)
            cv2.imwrite('results/%d.png'% j, (grad-mi)/(ma-mi)*255)

    def forward(self, x):
        feats = []
       
        for xi in x:
            feat_i = self.feat_extract(xi)
            atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            feats.append(feat_i * atten_i)
        
        # self.cal_feat_grad(feats)
        
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[9*j:9*(j+1)], dim=2)  # B*8*9*h*w
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2) # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))

        cv = torch.cat(feats_angle, 1)
        for j in range(3):
            cv = self.fuse2d[j*2+1](self.fuse2d[j*2](cv) + cv )
       
        # disp = self.fuse3(cv2)
        prob = self.relu3(self.fuse3(cv))
        disp = self.disparitygression(prob)
        # return pred + x[40].squeeze(1)
        return disp

    def disparitygression(self, input):
        N, D, H, W = input.shape
        disparity_values = torch.linspace(-4,4,9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = torch.tile(x, (N, 1, H, W))
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

'''
# v3
class LFattModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # if self.isTrain:
        #     self.model_names = ['G',  'D']
        # else:  # during test time, only load G
        self.model_names = ['EPI']
        # define networks (both generator and discriminator)
        net = UnsuperviseNetv3(opt,self.isTrain)
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views
        self.center_index = self.use_v // 2
        self.pad = opt.pad

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = Lossv3(self.opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            # self.optimizer = torch.optim.SGD(self.netEPI.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        
    
    def set_input(self, inputs, epoch):
        self.epoch = epoch
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
        
        # self.input= [innn.to(self.device) for innn in inputs[0]+inputs[1]+inputs[2]+inputs[3]]
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output, self.raw_warp_img = self.netEPI(self.input)  # G(A)
        self.test_loss_log = 0 # self.test_loss(self.output[:,:,self.pad:-self.pad,self.pad:-self.pad], self.label[:,:,self.pad:-self.pad, self.pad:-self.pad])
     

    def backward_G(self):
        self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        self.raw_loss = 0
        for views in self.raw_warp_img:
            self.raw_loss += self.cal_l1(views)  # 
        self.loss_total = 0.6*self.loss_L1 + 0.4*self.raw_loss
        # self.loss_smoothness = get_smooth_loss(self.output, self.center_input) 
        if 'smoothness' in self.loss_names and self.epoch > 80:
            self.loss_total +=  self.loss_smoothness
        # self.loss_smoothness = get_smooth_loss(self.output, self.center_input)
        # self.loss_total = self.loss_L1 + self.loss_smoothness 
        self.loss_total.backward()

    def cal_l1(self, x_list):
        loss = 0
        for j in range(2,7):          #####################################
            if self.pad > 0:
                loss += self.test_loss(x_list[j][:,:,self.pad:-self.pad, self.pad:-self.pad], x_list[self.center_index][:,:,self.pad:-self.pad, self.pad:-self.pad])
            else:
                loss += self.test_loss(x_list[j], x_list[self.center_index])
        return loss

    def optimize_parameters(self):
 
        self.netEPI.train()
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights

# v2 old
class LFattModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # if self.isTrain:
        #     self.model_names = ['G',  'D']
        # else:  # during test time, only load G
        self.model_names = ['EPI']
        # define networks (both generator and discriminator)
        net = UnsuperviseNetv2(opt,self.isTrain)
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views
        self.center_index = self.use_v // 2
        self.pad = opt.pad

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = Loss(self.opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            # self.optimizer = torch.optim.SGD(self.netEPI.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        
    
    def set_input(self, inputs, epoch):
        self.epoch = epoch
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
        
        # self.input= [innn.to(self.device) for innn in inputs[0]+inputs[1]+inputs[2]+inputs[3]]
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output = self.netEPI(self.input)  # G(A)
        self.test_loss_log = 0 # self.test_loss(self.output[:,:,self.pad:-self.pad,self.pad:-self.pad], self.label[:,:,self.pad:-self.pad, self.pad:-self.pad])
     

    def backward_G(self):
        self.loss_L1 = self.criterionL1(self.output, self.supervise_view[:,:,:,:,1:6,1:6])
        self.loss_total = self.loss_L1
        # self.loss_smoothness = get_smooth_loss(self.output, self.center_input) 
        # if 'smoothness' in self.loss_names and self.epoch > 80:
        #     self.loss_total +=  self.loss_smoothness
        self.loss_total.backward()


    def optimize_parameters(self):
        self.netEPI.train()
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights




 # v4

# v4
class LFattModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        if opt.losses.find('cosin') != -1:
            self.loss_names.append('cosin')

        self.visual_names = ['center_input', 'output','label']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # if self.isTrain:
        #     self.model_names = ['G',  'D']
        # else:  # during test time, only load G
        self.model_names = ['EPI']
        # define networks (both generator and discriminator)
        net = UnsuperviseNetv4(opt,self.isTrain)
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views
        self.center_index = self.use_v //2
        self.pad = opt.pad
        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
  
        if self.isTrain:
            # define loss functions
            self.criterionL1 = Loss(self.opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            # self.optimizer = torch.optim.SGD(self.netEPI.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        
    
    def set_input(self, inputs, epoch):
        self.epoch = epoch
        self.input= [innn.to(self.device) for innn in inputs[0]]
        self.center_input = inputs[0][self.center_index].to(self.device)
        self.label = inputs[4].to(self.device)
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output, self.atten2 = self.netEPI(self.input)  # G(A)
        self.test_loss_log =0 # self.test_loss(self.output[:,:,self.pad:-self.pad,self.pad:-self.pad], self.label[:,:,self.pad:-self.pad, self.pad:-self.pad])
     
    

    def backward_G(self):
        self.warpping(self.output, self.input, IDX[0])
        self.loss_total = self.loss_L1 

        # self.loss_smoothness = get_smooth_loss(self.output, self.center_input) 
        # if 'smoothness' in self.loss_names and self.epoch > 80:
        #     self.loss_total +=  self.loss_smoothness

        if 'cosin' in self.loss_names :
            self.loss_total +=  0.2*self.loss_cosin
        self.loss_total.backward()
    
    def warpping(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, W), torch.arange(0, H)
        meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(4-self.center_index, 5+self.center_index):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k-4+self.center_index], grid, align_corners=True))  

        self.loss_L1 = 0
        for j in range(self.use_v):
            if self.pad>0:
                self.loss_L1 += self.test_loss(tmp[j][:,:,self.pad:-self.pad, self.pad:-self.pad], tmp[self.center_index][:,:,self.pad:-self.pad,self.pad:-self.pad])
            else:
                self.loss_L1 += self.test_loss(tmp[j], tmp[self.center_index])
        views = torch.stack(tmp, 1) # B*C*H*W*9
        grad = torch.mean(torch.abs(views[:,1:,:,:,:]- views[:,:self.use_v-1,:,:,:]), dim=2) # B*8*H*W
        grad_mask = torch.where(grad<0.02, 1., 0.)
        grad_mask = torch.cat([grad_mask[:,:self.center_index,:,:], torch.ones((B,1,H,W)).to(self.device), grad_mask[:,self.center_index:,:,:]], 1)
        if self.pad>0:
            self.loss_cosin = torch.mean(torch.cosine_similarity(self.atten2[:,:,self.pad:-self.pad,self.pad:-self.pad], grad_mask[:,:,self.pad:-self.pad,self.pad:-self.pad], dim=1))
        else:
            self.loss_cosin = torch.mean(torch.cosine_similarity(self.atten2, grad_mask, dim=1))
       
    def cal_l1(self, x_list):
        loss = 0
        for j in range(self.use_v):
            if self.pad>0:
                loss += self.test_loss(x_list[j][:,:,self.pad:-self.pad, self.pad:-self.pad], x_list[self.center_index][:,:,self.pad:-self.pad,self.pad:-self.pad])
            else:
                loss += self.test_loss(x_list[j], x_list[self.center_index])
        return loss

    def optimize_parameters(self):
        """
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        """
        self.netEPI.train()
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights
 # v3
'''        

'''  # 3dunet newv2
class LFattModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # if self.isTrain:
        #     self.model_names = ['G',  'D']
        # else:  # during test time, only load G
        self.model_names = ['EPI']
        # define networks (both generator and discriminator)
        net = UnsuperviseNetv2(opt,self.isTrain)
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views
        self.center_index = self.use_v // 2
        self.pad = opt.pad

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = Lossv2(self.opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            # self.optimizer = torch.optim.SGD(self.netEPI.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        
    
    def set_input(self, inputs, epoch):
        self.epoch = epoch
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
        
        # self.input= [innn.to(self.device) for innn in inputs[0]+inputs[1]+inputs[2]+inputs[3]]
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
        
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.output, self.lossmask = self.netEPI(self.input)  # G(A)
        else:
            self.output, _ = self.netEPI(self.input)  # G(A)
        self.test_loss_log = 0 # self.test_loss(self.output[:,:,self.pad:-self.pad,self.pad:-self.pad], self.label[:,:,self.pad:-self.pad, self.pad:-self.pad])
     

    def backward_G(self):
        # self.loss_L1 = self.criterionL1(self.input[0:18], self.output[1], self.output[2], self.output[3], self.output[4])
        if self.epoch < 40:
            self.loss_L1 = self.criterionL1(self.input[0:18], self.output)
        else:
            self.loss_L1 = self.criterionL1(self.input[0:18], self.output, self.lossmask)
        self.loss_total = self.loss_L1
        self.loss_smoothness = get_smooth_loss(self.output[0], self.center_input) 
        if 'smoothness' in self.loss_names and self.epoch > 90:
            self.loss_total +=  0.1*self.loss_smoothness
        self.loss_total.backward()



    def optimize_parameters(self):
        self.netEPI.train()
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights
'''


class Feature1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature1, self).__init__()
        pad = kernel_size // 2 
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
       

        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=8))
       
        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(16,out_channels,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature, self).__init__()
        pad = kernel_size // 2 
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
       
        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=8))
       
        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(16,out_channels,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x

class Feature9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature9, self).__init__()
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 4, (1, 3,3),stride=1, padding=(0,1,1)), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)

        self.branch1 = nn.Sequential(nn.AvgPool2d((1,2,2),), conv_2d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,2,2)))
        self.branch2 = nn.Sequential(nn.AvgPool2d((1,4,4)), conv_3d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,4,4)))
        self.branch3 = nn.Sequential(nn.AvgPool2d((1,8,8)), conv_3d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,8,8)))
        
        self.lastconv = nn.Conv3d(28,out_channels,(1, 3,3),stride=1, padding=(0,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_3d(in_c, out_c) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample= downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = torch.cat([x, self.branch1(x), self.branch2(x), self.branch3(x)], 1)
        x = self.lastconv(x)
        return x

class Feature36(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature36, self).__init__()
        
        # pad = kernel_size // 2 
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 4, (1, 3,3),stride=1, padding=(0,1,1)), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)

        self.branch1 = nn.Sequential(nn.AvgPool3d((1,2,2),), conv_3d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,2,2)))
        self.branch2 = nn.Sequential(nn.AvgPool3d((1,4,4)), conv_3d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,4,4)))
        self.branch3 = nn.Sequential(nn.AvgPool3d((1,8,8)), conv_3d(16, 4), nn.LeakyReLU(0.2, True), nn.Upsample(scale_factor=(1,8,8)))
        
        self.lastconv = nn.Conv3d(28,out_channels,(1, 3,3),stride=1, padding=(0,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_3d(in_c, out_c) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample= downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = torch.cat([x, self.branch1(x), self.branch2(x), self.branch3(x)], 1)
        x = self.lastconv(x)
        return x
