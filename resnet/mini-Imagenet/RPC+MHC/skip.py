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

class reshaped_convolution(nn.Module):
    def __init__(self, inchannels, outchannels, down_size, kernel_size, mode = 'RPC', times = 4):
        super(reshaped_convolution, self).__init__()
        self.last_size = 7
        self.inp = inchannels
        self.outp = outchannels
        self.ds = down_size
        self.size = kernel_size
        self.mode = mode
        self.times = times
        self.reshape = nn.Sequential(
            #nn.BatchNorm2d(self.inp*self.times),
            #nn.ReLU(),
            nn.Conv2d(self.inp*self.times, self.inp, self.size, 1, self.size//2, groups=self.inp, bias=False),
            nn.BatchNorm2d(self.inp),
            nn.ReLU(),
            nn.Conv2d(self.inp, self.outp, 1, 1, bias=False),
            nn.BatchNorm2d(self.outp),
            nn.ReLU()
        )

    def forward(self, x):
        if self.mode == 'RPC':
            x = self.get_newsz(x)
            out = self.reshape(x)
            return out
        if self.mode == 'ACH':
            x = self.get_newsz(x)
            out = self.reshape(x)
            return out

    def get_newsz(self, x):
        if self.mode == 'RPC':
            B, C, H, W = x.shape
            x = x.view(B, -1, H//self.ds, W//self.ds)
            return x
        if self.mode == 'ACH':
            B, C, H, W = x.shape
            times = H//self.last_size
            self.times = times**2
            x = x.view(B, -1, self.last_size, self.last_size)
            return x


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
            nn.BatchNorm2d(in_channel*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=3 // 2,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        #self.se = SeModule(out_channel)
        # self.cse = cse(out_channel, out_channel, SeModule(out_channel))

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

        # out = self.cse(out)
   #############################################
        '''
        op = torch.nn.AdaptiveAvgPool2d(int(0.6*H))
        out1 = op(x)
        out1 = F.interpolate(out1, [H, W])
        out1 = torch.cat([x, out1], dim = 1)
        out1 = self.SA(out1)
        if self.use_se == True:
            out += out1
        '''
   #############################################
        out += identity

        out = self.relu(out)
        #out_cse = self.cse(out)
        #return out_cse
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
        #self.cse = cse(out_channel, out_channel, SeModule(out_channel))
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
            out = out + out1#在这里不out = conv(RP + conv1()), 就是说直接用RP代替stride == 2的情况
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
        self.reshape_1 = reshaped_convolution(512, 512, 2, 3, 'ACH', 16)
        self.reshape_2 = reshaped_convolution(1024, 1024, 2, 3, 'ACH', 4)
        self.last_conv = nn.Sequential(
            nn.Conv2d(3584, 2048, 1, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #self.fc = nn.Linear(768 * block.expansion, num_classes)

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
        out1 = self.reshape_1(x)
        x = self.layer3(x)
        out2 = self.reshape_2(x)
        x = self.layer4(x)
        out = torch.cat([x, out1, out2], dim=1)
        x = self.last_conv(out)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x



def resnet34(num_classes=10, include_top=True):
    return ResNet(BasicBlock,[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)




