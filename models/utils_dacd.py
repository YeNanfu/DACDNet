import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


# apply spatial mask
def apply_spatial_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
        # print(mask)
    print("mask.shape:",mask.shape)
    print("x.shape:",x.shape)
    return x * mask

class Masker_spatial(nn.Module):
    def __init__(self, in_channels, mask_channel_group=1, mask_size=None):
        super(Masker_spatial, self).__init__()
        self.mask_channel_group = mask_channel_group
        self.mask_size = mask_size
        self.conv = conv1x1(in_channels, mask_channel_group*2,bias=True)
        self.conv_flops_pp = self.conv.weight.shape[0] * self.conv.weight.shape[1] + self.conv.weight.shape[1]
        self.conv.bias.data[:mask_channel_group] = 5.0
        self.conv.bias.data[mask_channel_group+1:] = 0.0
        # self.feature_size = feature_size
        # self.expandmask = ExpandMask(stride=dilate_stride, padding=1, mask_channel_group=mask_channel_group)
    

    def forward(self, x, temperature):
        mask =  F.adaptive_avg_pool2d(x, self.mask_size)
        flops = mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        mask = self.conv(mask)
        flops += self.conv_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c,h,w = mask.shape
        mask = mask.view(b,2,c//2,h,w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.mean()
        # print('spatial mask:')
        # print(mask)
        # print('spatial mask sparsity:', sparsity)
        return mask, sparsity, flops  


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, mask_channel_group=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        self.mask_channel_group = mask_channel_group
        
    def forward(self, x):
        if self.stride > 1:
            self.pad_kernel = torch.zeros((self.mask_channel_group,1,self.stride, self.stride), device=x.device)
            self.pad_kernel[:,:,0,0] = 1
            
            # print(f'self.pad_kernel: {self.pad_kernel}')

        self.dilate_kernel = torch.ones((self.mask_channel_group, self.mask_channel_group, 1 + 2 * self.padding, 1 + 2 * self.padding), device='cpu')        
        # print(f'self.dilate_kernel: {self.dilate_kernel}')
        
        x = x.float()
        
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
        
        