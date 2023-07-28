import torch.nn as nn
import torch
import torch.functional as F


class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_2d, self).__init__()
        pad = kernel_size // 2 
        # self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x) 

class conv_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_3d, self).__init__()
        pad = kernel_size // 2 
        block = []
        block.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad))
        block.append(nn.BatchNorm3d(out_channels))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        return self.conv(x) 

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  dilation,  downsample=None):
        super(conv_block, self).__init__()
        # pad = kernel_size // 2 
        
        self.downsample = downsample
        block = nn.ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2, True))
        block.append(nn.Conv2d(out_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x)
        return self.conv(x) + x_skip

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
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x

def default_conv(in_channels, out_channels, kernel_size, bias= False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## conv-bn-relu
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out



## x-conv-bn-conv-bn-relu
##  --------------------^
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## dense block
## bn-relu-1*1-bn-relu-3*3
class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
class _DenseBlock(nn.Sequential):
    """DenseBlock
       growth_rate: 增长率， outchannel = num_input_features + (num_layers-1) * growth_rate
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


## OctaveConv
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        
        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)
        
        X_h = X_h2h + X_l2h

        return X_h

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


from einops import rearrange
import math

"""

input  : B*C*A*H*W

output : B*C*A*H*W

"""

class transformer_layer(nn.Module):
    def __init__(self,channels,angles,layer_num,num_heads):
        super().__init__()
        self.channels = channels
        self.angRes = angles

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = num_heads
        self.MHSA_params['dropout'] = 0.

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AngTrans(self.channels,self.angRes,self.MHSA_params))
        return nn.Sequential(*layers)

    def forward(self, lr):

        # [B, C(hannels), A, h, w]
        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)

        buffer = lr

        # Position Encoding
        ang_position = self.pos_encoding(buffer, dim=2, token_dim=self.channels)
        for m in self.modules():
            m.ang_position = ang_position

        # Alternate AngTrans & SpaTrans
        out = self.altblock(buffer) + buffer

        return out


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim, token_dim):
        self.token_dim = token_dim # 8
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim

        pos_size = [1, 1, 1, 1, 1, self.token_dim]
        length = x.size(dim)
        pos_size[dim] = length

        pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
        pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
        pos_dim = pos_dim.view(pos_size)

        position = pos_dim

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes #9
        self.ang_dim = channels #8
        self.norm = nn.LayerNorm(self.ang_dim) #8 
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer

if __name__ == '__main__':

    model = transformer_layer(8,9,1,4)
    x = torch.randn((4,8,9,64,64))
    y = model(x)
    print()
   