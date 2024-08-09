import sys
import os
sys.path.append(os.path.dirname(__file__)) 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def ps(): 
    res = []
    wd = os.getcwd()
    res.append(wd)
    for root, dirs, files in os.walk(os.getcwd()):
        for dir in dirs:
            dir_path = os.path.join(root,dir)
            if dir_path[-1] != '_':                
                res.append(dir_path)   
    sys.path.extend(res)
ps()
# import sys
# print('sys.path:',sys.path)
from imagenet_classification.models.utils_dacd import conv1x1, conv3x3, Masker_spatial, ExpandMask, apply_spatial_mask

class Cacd_Bottleneck(nn.Module):
    expansion = 2  

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None,
                 output_size=None, 
                 mask_spatial_granularity=None):
        super(Cacd_Bottleneck, self).__init__()
            
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        base_width = 64
        width = int(planes * (base_width / 64.)) * group_width  # 等于planes
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group_width, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        
        self.conv1_flops_per_pixel = inplanes*width
        self.conv2_flops_per_pixel = width*width*9 // self.conv2.groups  
        self.conv3_flops_per_pixel = width*planes*self.expansion

        if self.downsample is not None:  
            self.downsample_flops = inplanes * planes * self.expansion

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity  # 图片是方形的所以可以用这种简单的除法算掩码的大小
        self.mask_size = torch.tensor([self.output_size[0] // self.mask_spatial_granularity[0], self.output_size[1] // self.mask_spatial_granularity[1]])

        self.masker_spatial = None
        

        self.masker_spatial = Masker_spatial(inplanes, mask_size=self.mask_size)
        self.mask_expander2 = ExpandMask(stride=1, padding=0)  
        self.mask_expander1 = ExpandMask(stride=stride, padding=1)
             

    def forward(self, x, temperature=1.0):
        
        x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, flops_perc_list, flops = x 
        identity = x
        
        spatial_mask_conv3, spatial_sparsity_conv3, spatial_mask_flops = self.masker_spatial(x, temperature)

  
        spatial_mask_conv3 = F.interpolate(spatial_mask_conv3, size=self.output_size, mode='nearest')
        spatial_mask_conv2 = self.mask_expander2(spatial_mask_conv3)
        spatial_sparsity_conv2 = spatial_mask_conv2.float().mean()
        spatial_mask_conv1 = self.mask_expander1(spatial_mask_conv2)
        spatial_sparsity_conv1 = spatial_mask_conv1.float().mean()
        
        sparse_flops = spatial_mask_flops
        dense_flops = spatial_mask_flops
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]  * spatial_sparsity_conv1
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]  * spatial_sparsity_conv2
        
        out = self.conv3(out)
        out = self.bn3(out)
        print('out.shape:',out.shape)
        out = apply_spatial_mask(out, spatial_mask_conv3)
        
        dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]  * spatial_sparsity_conv3
        
        if self.downsample is not None:
            identity = self.downsample(x)
            dense_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
            sparse_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
        
        out += identity
        out = self.relu(out)

        flops += sparse_flops
        flops_perc = sparse_flops / dense_flops

        spatial_sparsity_conv3_list = spatial_sparsity_conv3.unsqueeze(0) if spatial_sparsity_conv3_list is None else \
            torch.cat((spatial_sparsity_conv3_list,spatial_sparsity_conv3.unsqueeze(0)), dim=0)
        
        spatial_sparsity_conv2_list = spatial_sparsity_conv2.unsqueeze(0) if spatial_sparsity_conv2_list is None else \
            torch.cat((spatial_sparsity_conv2_list,spatial_sparsity_conv2.unsqueeze(0)), dim=0)
        
        spatial_sparsity_conv1_list = spatial_sparsity_conv1.unsqueeze(0) if spatial_sparsity_conv1_list is None else \
            torch.cat((spatial_sparsity_conv1_list,spatial_sparsity_conv1.unsqueeze(0)), dim=0)
        
          
        flops_perc_list = flops_perc.unsqueeze(0) if flops_perc_list is None else \
            torch.cat((flops_perc_list,flops_perc.unsqueeze(0)), dim=0)
        
        return out, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, flops_perc_list, flops

class Cacd_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=6, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 input_size=[(100,3),(50,2),(25,1)], 
                 mask_spatial_granularity=[(5,1),(5,1),(5,1)],    
                 lr_mult=1.0,
                 **kwargs):
        super(Cacd_ResNet, self).__init__()
        assert lr_mult is not None
        self.lr_mult = lr_mult

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(16)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.layer1 = self._make_layer(block, int(16), layers[0], stride=1,
                                       dilate=False,
                                       output_size=input_size[0],
                                       mask_spatial_granularity=mask_spatial_granularity[0])
        
        self.layer2 = self._make_layer(block, int(32), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       output_size=input_size[1],
                                       mask_spatial_granularity=mask_spatial_granularity[1])
        
        self.layer3 = self._make_layer(block, int(48), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       output_size=input_size[2],
                                       mask_spatial_granularity=mask_spatial_granularity[2])
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48* block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'masker' not in name:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Cacd_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    output_size=None,
                    mask_spatial_granularity=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, group_width=self.groups,
                            dilation=previous_dilation, norm_layer=norm_layer, 
                            output_size=output_size,
                            mask_spatial_granularity=mask_spatial_granularity))
        self.inplanes = planes * block.expansion
        for j in range(1, blocks): 
            layers.append(block(self.inplanes, planes, group_width=self.groups,
                                dilation=self.dilation,
                                norm_layer=norm_layer, 
                                output_size=output_size,
                                mask_spatial_granularity=mask_spatial_granularity))

        return nn.ModuleList(layers)

    def forward(self, x, temperature = 1.0):  
        c_in = x.shape[1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # self.conv1.weight.shape = (out_channels, in_channels, kernel_height, kernel_width)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]

        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        
        spatial_sparsity_conv1_list, spatial_sparsity_conv2_list, spatial_sparsity_conv3_list, \
             flops_perc_list = None, None, None, None
        
        x = (x, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, flops_perc_list, flops)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x, temperature)
        x, spatial_sparsity_conv3_stage1, spatial_sparsity_conv2_stage1, spatial_sparsity_conv1_stage1, flops_perc_list, flops = x
        
        x = (x, None, None, None, flops_perc_list, flops)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, temperature)
        x, spatial_sparsity_conv3_stage2, spatial_sparsity_conv2_stage2, spatial_sparsity_conv1_stage2, flops_perc_list, flops = x
        
        x = (x, None, None, None, flops_perc_list, flops)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x, temperature)
        x, spatial_sparsity_conv3_stage3, spatial_sparsity_conv2_stage3, spatial_sparsity_conv1_stage3, flops_perc_list, flops = x
           
        x = self.avgpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]
        
        x = torch.flatten(x, 1)
        
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in*x.shape[1]
        
        spatial_sparsity_conv3 = [spatial_sparsity_conv3_stage1, spatial_sparsity_conv3_stage2, spatial_sparsity_conv3_stage3]
        spatial_sparsity_conv2 = [spatial_sparsity_conv2_stage1, spatial_sparsity_conv2_stage2, spatial_sparsity_conv2_stage3]
        spatial_sparsity_conv1 = [spatial_sparsity_conv1_stage1, spatial_sparsity_conv1_stage2, spatial_sparsity_conv1_stage3]
        
        return x, spatial_sparsity_conv3, spatial_sparsity_conv2, spatial_sparsity_conv1, flops_perc_list, flops
    
    def get_optim_policies(self):
        backbone_params = []
        masker_params = []

        for name, m in self.named_modules():
            if 'masker' in name:
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    masker_params.append(ps[0]) 
                    if len(ps) == 2:
                        masker_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    masker_params.extend(list(m.parameters()))  
                elif isinstance(m, torch.nn.BatchNorm1d):
                    masker_params.extend(list(m.parameters()))  
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    masker_params.append(ps[0])
                    if len(ps) == 2:
                        masker_params.append(ps[1])
            else:
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    backbone_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0])
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
        return [
            {'params': backbone_params, 'loptimizer = torch.optim.SGD(optim_policies, lr=0.01)r_mult': self.lr_mult, 'decay_mult': 1.0, 'name': "backbone_params"},
            {'params': masker_params, 'lr_mult': 1.0, 'decay_mult': 1.0, 'name': "masker_params"},
        ]

def _cacdresnet(arch, block, layers, **kwargs):
    model = Cacd_ResNet(block, layers, **kwargs)
    return model

def cacd_resnet10(pretrained=False, progress=True, **kwargs):
    print('Model: cacd_eesnet 10')
    return _cacdresnet('cdcd_resnet10', Cacd_Bottleneck, [1, 1, 1], 
                   **kwargs)


# test code
if __name__ == '__main__':
    from pthflops import count_ops

    wisdm_input_size = (1, 1, 200, 3) 
    # unimib_input_size = (1, 1, 151, 3) 
    # pamap2_input_size = (1, 1, 171, 30)

    x_wisdm = torch.randn(wisdm_input_size)
    model_wisdm = cacd_resnet10(num_classes=6)
    model_wisdm.eval()
    with torch.no_grad():
        print("wisdm_model is running...")
        y_wisdm = model_wisdm(x_wisdm)