import torch
import torch.nn as nn

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

class Resnext(nn.Module):
    def __init__(self, num_classes, layer=[3, 4, 23, 3]):
        super(Resnext, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, 256, 1, num=layer[0])
        self.conv3 = self._make_layer(256, 512, 2, num=layer[1])
        self.conv4 = self._make_layer(512, 1024, 2, num=layer[2])
        self.conv5 = self._make_layer(1024, 2048, 2, num=layer[3])
        self.global_average_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        #########MHC#########MHC#########MHC#########MHC#########MHC#########MHC#########MHC
        self.RPC1 = reshaped_convolution(512, 512, 2, 3, mode='ACH', times=16)
        self.RPC2 = reshaped_convolution(1024, 1024, 2, 3, mode='ACH', times=4)
        #########MHC#########MHC#########MHC#########MHC#########MHC#########MHC#########MHC
        self.last_conv = nn.Sequential(
            nn.Conv2d(3584, 2048, 1, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #########MHC#########MHC#########MHC#########MHC#########MHC
        first = self.RPC1(x)
        #########MHC#########MHC#########MHC#########MHC#########MHC
        x = self.conv4(x)
        #########MHC#########MHC#########MHC#########MHC#########MHC
        second = self.RPC2(x)
        #########MHC#########MHC#########MHC#########MHC#########MHC
        x = self.conv5(x)
        x_ = torch.cat((first, x, second), dim=1)
        x = self.last_conv(x_)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, out_channels, stride, num):
        layers = []
        block_1 = Block(in_channels, out_channels, stride=stride, is_shortcut=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(out_channels, out_channels, stride=1, is_shortcut=False))
        return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_shortcut=False):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.is_shortcut = is_shortcut
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1, groups=32,
                      bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        #########RPC#########RPC#########RPC#########RPC#########RPC#########RPC#########RPC
        self.reshape = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, kernel_size=3, stride=1, padding=3//2, groups=out_channels // 2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
        )
        #########RPC#########RPC#########RPC#########RPC#########RPC#########RPC#########RPC
        if is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=1),
                nn.BatchNorm2d(out_channels)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        #########RPC#########RPC#########RPC#########RPC#########RPC
        if self.stride == 2:
            B, C, H, W = x.shape
            x_ = x.view(B, -1, H//2, W//2)
            out = self.reshape(x_)
            x = self.conv2(x)
            x = x + out
        #########RPC#########RPC#########RPC#########RPC#########RPC
        else:
            x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x

def resNext(pretrained=False, progress=True, **kwargs):
    model = Resnext(100)
    return model