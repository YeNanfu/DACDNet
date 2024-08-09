import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 2
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups 
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)  
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # print(out.shape)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
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
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0])  
        self.layer3 = self._make_layer(block, 48, layers[2], stride=2, 
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        print('layers0: ', layers)
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        print('layers1: ', layers)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):  # 我们这里blocks[i]=1,所以这个循环不会执行
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        print('layers2: ', layers)
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        print("input: ", x.shape)
        x = self.conv1(x)
        print("after_conv1: ", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("after_maxpool: ", x.shape)
        x = self.layer1(x)
        print("after_layer1: ", x.shape)
        x = self.layer2(x)
        print("after_layer2: ", x.shape)
        x = self.layer3(x)
        print("after_layer3: ", x.shape)
        x = self.avgpool(x)
        print("after_avgpool: ", x.shape)
        x = torch.flatten(x, 1)
        print("after_flatten: ", x.shape)
        x = self.fc(x)
        print("after_fc: ", x.shape)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet10(num_classes):
    return _resnet('resnet10', Bottleneck, [1, 1, 1],num_classes=num_classes, pretrained = False, progress=True)




if __name__ == '__main__':

    from pthflops import count_ops

    # uci_input_size = (1, 1, 128, 9)  6
    # wisdm_input_size = (1, 1, 200, 3)  6
    # unimib_input_size = (1, 1, 151, 3)  17
    # pamap2_input_size = (1, 1, 171, 30) 12
    # usc_input_size = (1, 1, 512, 6)
    input_size = (1, 1, 200, 3)

    x = torch.randn(input_size)

    model = resnet10(num_classes=12)
    model.eval()

    with torch.no_grad():
        print("model is running...")
        y_b = model(x)

    flops, _ = count_ops(model, x)
    print(f'model has {flops/1000/1000}M FLOPs.')

