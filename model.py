import torch
import torch.nn as nn
import torch.nn.functional as F

def norm2d(out_channels, group, GN):
    if GN:
        return nn.GroupNorm(group, out_channels)
    else:
        return nn.BatchNorm2d(out_channels)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None, GN=False, group=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm2d(out_channels, group, GN)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm2d(out_channels, group, GN)
        self.downsample = downsample
        
        if downsample is not None:
            residual = []
            residual += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            residual += [norm2d(out_channels, group, GN)]
            self.downsample = nn.Sequential(*residual)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_class=10, layer=56, GN=False, group=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = norm2d(16, group, GN)
        self.relu = nn.ReLU(inplace=True)

        if layer is 14:
            self.n = 2
        elif layer is 20:
            self.n = 3
        elif layer is 32:
            self.n = 5
        elif layer is 44:
            self.n = 7
        elif layer is 56:
            self.n = 9
        elif layer is 110:
            self.n = 18
            

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, GN=GN, group=group))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, GN=GN, group=group))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True, GN=GN, group=group))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None, GN=GN, group=group))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True, GN=GN, group=group))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None, GN=GN, group=group))

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x