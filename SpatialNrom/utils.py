import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from model import *
import csv
import random
import os
import math
import numbers

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'

class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self, args, num_class=10):
        layer = [14, 20, 32, 44, 56, 110]
        self.model = ResNet(num_class=num_class, layer=layer[args.layer], Method=args.Method, Mode=args.Mode, norm_type=args.norm_type)

    def forward(self, x, test=False):
        output = self.model(x)
        return output

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model).cuda()

    def weight_init(self):
        self.model.apply(self.weights_init_normal)

    def weights_init_normal(self, m, init_type="kaiming"):


        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, args):
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def set_train_mode(self):
        self.model.train()

    def set_test_mode(self):
        self.model.eval()

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['dict'])
        self.optimizer.load_state_dict(checkpoint['optim'])


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_state_checkpoint(state_info, best_prec_result, filename, directory, epoch):
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,

        'model': state_info.model,
        'dict': state_info.model.state_dict(),
        'optim': state_info.optimizer.state_dict(),

    }, filename, directory)

def save_checkpoint(state, filename, model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(directory, is_last=True):

    if is_last:
        load_state_name = os.path.join(directory, 'latest.pth.tar')
    else:
        load_state_name = os.path.join(directory, 'checkpoint_best.pth.tar')
    
    if os.path.exists(load_state_name):
        print("=> loading checkpoint '{}'".format(load_state_name))
        state = torch.load(load_state_name)
        return state
    else:
        return None

def print_log(text, filename="log.csv"):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    model_filename = os.path.join(default_model_dir, filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")