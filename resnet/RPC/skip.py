import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out



class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.se(x)
        last_out = x * out
        return last_out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, use_sa = True, downsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.use_se = use_sa
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.SA = nn.Sequential(
                   nn.Conv2d(in_channels=2*in_channel, out_channels=2*in_channel,
                               kernel_size=3, stride=1, padding=1, groups = 2*in_channel, bias=False),
                   nn.BatchNorm2d(2*in_channel),
                   nn.ReLU(),

                   nn.Conv2d(in_channels=2*in_channel, out_channels=out_channel,
                               kernel_size=3, stride=self.stride, padding=1,  bias=False),
                   nn.BatchNorm2d(out_channel),
                   nn.ReLU(),
        )
        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=4 * in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=3 // 2,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x
        B, C, W, H = x.shape
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv0(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride == 2:
            x = x.view(B, -1, H//2, W//2)
            out1 = self.reshape(x)
            out = out + out1





        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_sa = False):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.reshape = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*4, out_channels=out_channel, kernel_size=3, stride=1, padding=3//2, groups=out_channel, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

        )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride == 1:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
        elif self.stride == 2:
            B, C, H, W = out.shape
            out_ = out.view(B, -1, H//2, W//2)
            out1 = self.reshape(out_)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = out + out1
        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=10, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], use_sa = True)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], use_sa = True, stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], use_sa = False, stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], use_sa = False, stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, use_sa = True, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, use_sa = use_sa))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,use_sa = use_sa))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=10, include_top=True):
    return ResNet(BasicBlock,[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
def resnet34(num_classes=10, include_top=True):
    return ResNet(BasicBlock,[2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    
def resnet101(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
    
def resnet152(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)





