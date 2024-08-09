import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础3x3卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 基础1x1卷积，用于升维或降维
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

# 应用通道掩码
def apply_channel_mask(x, mask):
    b, c, h, w = x.shape
    _, g = mask.shape
    if (g > 1) and (g != c):
        mask = mask.repeat(1,c//g).view(b, c//g, g).transpose(-1,-2).reshape(b,c,1,1)
    else:
        mask = mask.view(b,g,1,1)
    return x * mask

# 应用空间掩码
# TODO: 掩码的hw需要针对har优化，这里考虑之后不优化了，没有必要
def apply_spatial_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
        # print(mask)
    return x * mask

# 图三(a)用于生成空间掩码,用1x1卷积实现，
# 先用adaptive_avg_pool2d将特征图缩小成需要的掩码大小，然后用1x1卷积输出掩码，最后用gumbel_softmax输出掩码，返回掩码、稀疏度和FLOPs
# 最后用gumbel_softmax输出掩码，返回掩码、稀疏度和FLOPs
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
    
    # 前向传播，同时计算FLOPs
    def forward(self, x, temperature):
        mask =  F.adaptive_avg_pool2d(x, self.mask_size)
        flops = mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        mask = self.conv(mask)
        flops += self.conv_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c,h,w = mask.shape
        mask = mask.view(b,2,c//2,h,w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)  # 用gumbel_softmax输出掩码
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.mean()
        # print('spatial mask:')
        # print(mask)
        # print('spatial mask sparsity:', sparsity)
        return mask, sparsity, flops  # 返回掩码、稀疏度和FLOPs

# 用于扩张掩码,扩张到特征图大小，用conv2d实现
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

        self.dilate_kernel = torch.ones((self.mask_channel_group,self.mask_channel_group,1+2*self.padding,1+2*self.padding), device=x.device)
        # print(f'self.dilate_kernel: {self.dilate_kernel}')
        
        x = x.float()
        
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5

# 图三(b)用于生成通道掩码，用全连接实现，
# 先用adaptive_avg_pool2d将特征图缩小成1x1，然后用全连接输出掩码，
# 最后用gumbel_softmax输出掩码，返回掩码、稀疏度和FLOPs
class Masker_channel_MLP(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, layers=2, reduction=16):
        super(Masker_channel_MLP, self).__init__()
        assert(layers in [1,2])  # layers=1表示只有一层全连接，layers=2表示有两层全连接
        
        self.channel_dyn_group = channel_dyn_group
        width = max(channel_dyn_group//reduction, 16)  # 用于降维
        self.conv = nn.Sequential(  
            nn.Linear(in_channels, width),
            nn.ReLU(),
            nn.Linear(width, channel_dyn_group*2,bias=True)  # 
        ) if layers == 2 else nn.Linear(in_channels, channel_dyn_group*2,bias=True)  # layers=2表示有两层全连接，layers=1表示只有一层全连接
        
        self.conv_flops = in_channels * width + width * channel_dyn_group*2 if layers == 2 else in_channels * channel_dyn_group*2  # 计算FLOPs
        if layers == 2:  # 最后一层全连接层的bias初始化，前channel_dyn_group个为2，后channel_dyn_group个为-2，这种初始化方式是为了让掩码的初始值接近0.5
            self.conv[-1].bias.data[:channel_dyn_group] = 2.0  
            self.conv[-1].bias.data[channel_dyn_group+1:] = -2.0  
        else:
            self.conv.bias.data[:channel_dyn_group] = 2.0
            self.conv.bias.data[channel_dyn_group+1:] = -2.0

    def forward(self, x, temperature):
        b, c, h, w = x.shape
        flops = c * h * w
        mask =  F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        
        mask = self.conv(mask)
        flops += self.conv_flops
        
        b,c = mask.shape
        mask = mask.view(b,2,c//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        
        return mask, sparsity, flops

# 这个计算量和两层全连接相比要小很多，但是效果不好
class Masker_channel_conv_linear(nn.Module):
    def __init__(self, in_channels, channel_dyn_group, reduction=16):
        super(Masker_channel_conv_linear, self).__init__()
        self.channel_dyn_group = channel_dyn_group
        
        self.conv = nn.Sequential(
            conv1x1(in_channels, in_channels//reduction), # 1x1卷积,用于降维
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(),
        )
        self.linear = nn.Linear(in_channels//reduction, channel_dyn_group*2,bias=True)
        
        self.linear.bias.data[:channel_dyn_group] = 2.0
        self.linear.bias.data[channel_dyn_group+1:] = -2.0
        
        self.masker_flops = in_channels * in_channels // reduction + in_channels // reduction * channel_dyn_group*2

    def forward(self, x, temperature):
        mask = self.conv(x)
        b, c, h, w = mask.shape
        flops = c * h * w
        mask =  F.adaptive_avg_pool2d(mask, (1,1)).view(b,c)
        
        mask = self.linear(mask)
        flops += self.masker_flops
        
        b,c = mask.shape
        mask = mask.view(b,2,c//2)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        
        sparsity = torch.mean(mask)
        
        return mask, sparsity, flops


if __name__ == '__main__':  # 测试代码，外部调用时不会执行
    with torch.no_grad():
        from pthflops import count_ops
        input_size = (1, 1, 151, 3)
        x = torch.randn(input_size)
        model_b = Masker_channel_MLP(in_channels=32, channel_dyn_group=2, layers=2, reduction=16,temperature=1)
        model_b.eval()

        with torch.no_grad():
            print("model_b is running...")
            y_b = model_b(x)

        # 计算模型的 FLOPs
        flops_usc_b, _ = count_ops(model_b, x)
        print(f'model_b has {flops_usc_b/1000/1000}M FLOPs.')
        
        