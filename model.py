import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os 

def print_log(text, filename="channel_sel.csv"):
    model_filename = os.path.join('./', filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")

def norm2d(out_channels, group, Method, batch_size=None, width_height=None, use=False):
    if not use:
        return nn.GroupNorm(group, out_channels)

    if Method == "GN":
        return nn.GroupNorm(group, out_channels)
    elif Method == "BN":
        return nn.BatchNorm2d(out_channels)
    elif Method == "P1":
        return Proposed_ver1(width_height, batch_size, group, out_channels)
    elif Method == "P2":
        return Proposed_ver2(width_height, group, out_channels)

def print_time(start_time, log):
    print('{} : {}'.format(log, time.time() - start_time))

def print_time_relay(start_time, log):
    now = time.time()
    print('{} : {}'.format(log, now - start_time))
    return now

class GroupNorm(nn.Module):
    def __init__(self, group, out_channels, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,out_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1))
        self.group = group
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.group

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

class Proposed_ver1(nn.Module):
    def __init__(self, width_height, batch_size, group, out_channels, eps=1e-5):
        super(Proposed_ver1, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,out_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1))
        self.group = group
        self.eps = eps
        self.model = nn.Sequential(
            nn.Conv2d(batch_size, batch_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=width_height, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(batch_size, batch_size),
            nn.Linear(batch_size, group),
        )
    def forward(self, x):
        N,C,H,W = x.size()
        x_ = torch.transpose(x,0,1) # transpose 후 x_.size() == [C,N,H,W]
        x_ = self.model(x_)
        x_ = x_.view(C, -1)
        s = F.softmax(self.fc(x_), dim=1) # s.view() == [C, group]
        _, s = torch.max(s.data, dim=1) # s.view() == [C], max Index data
        print_log(s.item())
        group_list = torch.FloatTensor(C, self.group).cuda().zero_().scatter_(1, s.view(-1, 1), 1).transpose(0,1)
        arr = []
        for i in range(self.group):
            arr.append([])
            arr[i] = torch.nonzero(group_list[i]).view(-1)
        for grp in arr:
            if len(grp) is 0:
                continue
            box = x[:,grp].view(N,-1)
            box_mean = box.mean(-1, keepdim=True).view(-1,1,1,1)
            box_var = box.var(-1, keepdim=True).view(-1,1,1,1)
            x[:,grp] = (x[:,grp] - box_mean) / (box_var + self.eps).sqrt()
        return x * self.weight + self.bias

# class Proposed_ver1(nn.Module):
#     def __init__(self, batch_size, group, out_channels, eps=1e-5):
#         super(Proposed_ver1, self).__init__()
#         self.weight = nn.Parameter(torch.ones(1,out_channels,1,1))
#         self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1))
#         self.group = group
#         self.eps = eps
#         self.fc = nn.Linear(batch_size * 2, group)
#     def forward(self, x):
#         N,C,H,W = x.size()
#         start_time = time.time()
#         x_ = torch.transpose(x,0,1).view(C, N, -1) # transpose 후 x_.size() == [C,N,H,W]
#         start_time = print_time_relay(start_time, "P1_1s duration")
#         mean = x_.mean(-1, keepdim=True).squeeze(-1) # mean.size() == [C,N]
#         start_time = print_time_relay(start_time, "P1_2s duration")
#         var = x_.var(-1, keepdim=True).squeeze(-1) # var.size() == [C,N]
#         start_time = print_time_relay(start_time, "P1_3s duration")
#         s = torch.cat([mean, var], 1) # s.view() == [C, 2N]
#         start_time = print_time_relay(start_time, "P1_4s duration")
#         s = F.softmax(self.fc(s), dim=1) # s.view() == [C, group]
#         start_time = print_time_relay(start_time, "P1_5s duration")
#         _, s = torch.max(s.data, dim=1) # s.view() == [C], max Index data
#         start_time = print_time_relay(start_time, "P1_6s duration")
#         group_list = torch.FloatTensor(C, self.group).cuda().zero_().scatter_(1, s.view(-1, 1), 1).transpose(0,1)
#         start_time = print_time_relay(start_time, "P1_7s duration")
#         arr = []
#         for i in range(self.group):
#             arr.append([])
#             arr[i] = torch.nonzero(group_list[i]).view(-1)
#         start_time = print_time_relay(start_time, "P1_8s duration")
#         for grp in arr:
#             if len(grp) is 0:
#                 continue
#             box = x[:,grp].view(N,-1)
#             box_mean = box.mean(-1, keepdim=True).view(-1,1,1,1)
#             box_var = box.var(-1, keepdim=True).view(-1,1,1,1)
#             x[:,grp] = (x[:,grp] - box_mean) / (box_var + self.eps).sqrt()
#         start_time = print_time_relay(start_time, "P1_9s duration")
#         return x * self.weight + self.bias

class Proposed_ver2(nn.Module):
    def __init__(self, width_height, group, out_channels, eps=1e-5):
        super(Proposed_ver2, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,out_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1))
        self.group = group
        self.eps = eps
        self.fc = nn.Sequential(
            nn.Linear(width_height ** 2, width_height ** 2),
            nn.Linear(width_height ** 2, group),
        )
    def forward(self, x):
        N,C,H,W = x.size()
        x_ = x.view(N*C, -1) # x_.size() == [C*N,H*W]
        s = F.softmax(self.fc(x_), dim=1) # s.view() == [C*N, group]
        _, s = torch.max(s.data, dim=1) # s.view() == [C*N], max Index data
        group_list = torch.FloatTensor(C*N, self.group).cuda().zero_().scatter_(1, s.view(-1, 1), 1).transpose(0,1)

        arr = []
        for i in range(self.group):
            arr.append([])
            arr[i] = torch.nonzero(group_list[i]).view(-1)
        for grp in arr:
            if len(grp) is 0:
                continue
            box = x_[grp].view(-1)
            box_mean = box.mean()
            box_var = box.var()
            x_[grp] = (x_[grp] - box_mean) / (box_var + self.eps).sqrt()
        x_ = x_.view(N,C,H,W)
        return x_ * self.weight + self.bias

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None, Method=False, group=1, batch_size=64, width_height=32, use=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm2d(out_channels, group, Method, batch_size=batch_size, width_height=width_height, use=use)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm2d(out_channels, group, Method, batch_size=batch_size, width_height=width_height, use=use)
        self.downsample = downsample
        
        if downsample is not None:
            residual = []
            residual += [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            residual += [norm2d(out_channels, group, Method, batch_size=batch_size, width_height=width_height)]
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
    def __init__(self, num_class=10, layer=56, Method=False, group=1, batch_size=None):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = norm2d(16, group, Method, batch_size=batch_size, width_height=32, use=True)
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
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, Method=Method, group=group, batch_size=batch_size, width_height=32))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, Method=Method, group=group, batch_size=batch_size, width_height=32))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True, Method=Method, group=group, batch_size=batch_size, width_height=16))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None, Method=Method, group=group, batch_size=batch_size, width_height=16))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True, Method=Method, group=group, batch_size=batch_size, width_height=8))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None, Method=Method, group=group, batch_size=batch_size, width_height=8))

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