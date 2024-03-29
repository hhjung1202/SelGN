import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os 

def norm2d(out_channels, Method, norm_type="batch", use=False):
    if not use:
        return nn.BatchNorm2d(out_channels)

    elif Method == "BN":
        return nn.BatchNorm2d(out_channels)
    elif Method == "P1":
        return SpatialNorm(out_channels, norm_type=norm_type)
    elif Method == "P2":
        return SpatialNorm2(out_channels, norm_type=norm_type)

class SpatialNorm(nn.Module):
    def __init__(self, channel, r=4, norm_type="batch", eps=1e-5, case=0):
        super(SpatialNorm, self).__init__()

        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(channel, affine=False)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(channel, affine=False)

        self.share = nn.Sequential(
            nn.Conv2d(channel, channel//r, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel//r),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        )

        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid(),
        )

        self.eps = eps
        self.case = case

    def forward(self, x):
        x = self.norm(x)

        x_out = self.share(x)
        gamma = self.conv_gamma(x_out)
        beta = self.conv_beta(x_out)

        print('all mean gamma',torch.mean(gamma))
        print('all mean beta',torch.mean(beta))

        print('all mean gamma2',torch.mean(gamma.view(gamma.size(0), -1), 1))
        print('all mean beta2',torch.mean(beta.view(beta.size(0), -1), 1))

        return x * (gamma + 0.5) + (beta-0.5)
        # if self.case is 0:
        #     return x * gamma + beta
        # elif  self.case is 1:
        #     return x * 2 * gamma + beta
        # elif  self.case is 2:
        #     return x * (gamma + 0.5) + (beta-0.5)
        # elif  self.case is 3:
        #     return x * gamma + (beta-0.5)


def getting(x):
    N,C,W,H = x.size()
    x_ = x.view(N,C,-1)
    _, sgmap = torch.max(x_, dim=1)
    sgmap = torch.FloatTensor(N,C,W*H).zero_().scatter_(1, sgmap.view(N, 1, -1), 1)
    s_sg = torch.sum(sgmap, dim=2, keepdim=True) + 1e-5
    print(sgmap)
    act_map = x_*sgmap
    print(act_map)
    s_map = torch.sum(act_map, dim=2, keepdim=True)
    print(s_map)
    mean = s_map/s_sg
    ss_map = torch.sum(act_map**2, dim=2, keepdim=True)
    var = ss_map/s_sg - mean**2
    print(mean**2)
    print(var)
    print(act_map*var.sqrt() + mean)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialNorm2(nn.Module):
    def __init__(self, channel, r=4, norm_type="batch", eps=1e-5):
        super(SpatialNorm2, self).__init__()

        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(channel, affine=False)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(channel, affine=False)

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(channel, channel//r),
            nn.BatchNorm1d(channel//r),
            nn.ReLU(True),
            nn.Linear(channel//r, channel),
            nn.Sigmoid(),
            UnFlatten(),
        )
        # kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2
        self.spatial = nn.Sequential(
            ChannelPool(),
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Sigmoid(),
        )

        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
        )

        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        x = self.norm(x)
        x_out = x * self.channel(x)
        x_out = x_out * self.spatial(x_out)

        gamma = self.conv_gamma(x_out)
        beta = self.conv_beta(x_out)

        print('all mean gamma',torch.mean(gamma))
        print('all mean beta',torch.mean(beta))

        print('all mean gamma2',torch.mean(gamma.view(gamma.size(0), -1), 1))
        print('all mean beta2',torch.mean(beta.view(beta.size(0), -1), 1))

        return x * (1 + gamma) + beta

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None, Method=False, use=False, norm_type="batch"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        # self.norm1 = norm2d(out_channels, Method, use=use, norm_type=norm_type)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm2d(out_channels, Method, use=use, norm_type=norm_type)
        self.downsample = downsample
        
        if downsample is not None:
            residual = []
            residual += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            residual += [nn.BatchNorm2d(out_channels)]
            # residual += [norm2d(out_channels, Method, use=use, norm_type=norm_type)]
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
    def __init__(self, num_class=10, layer=56, Method=False, Mode=0, norm_type="batch"):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = norm2d(16, Method, use=(Mode==0 or Mode==2), norm_type=norm_type)
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
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, Method=Method, use=(Mode==2), norm_type=norm_type))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, Method=Method, use=(Mode==2), norm_type=norm_type))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True, Method=Method, use=(Mode==2), norm_type=norm_type))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None, Method=Method, use=(Mode==2), norm_type=norm_type))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True, Method=Method, use=(Mode==2), norm_type=norm_type))
        for i in range(1,self.n-1):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None, Method=Method, use=(Mode==2), norm_type=norm_type))
        self.layer3.add_module('layer3_%d' % (self.n-1), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None, Method=Method, use=(Mode==1 or Mode==2), norm_type=norm_type))

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_class)

    def forward(self, x):

        print('start')

        x = self.conv1(x)
        x = self.norm(x)

        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        print('end')

        return x