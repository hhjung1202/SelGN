import argparse
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from model import *
import os
import torch.backends.cudnn as cudnn
import time
import utils
import dataset
import math


parser = argparse.ArgumentParser(description='PyTorch Cycle Domain Adaptation Training')
parser.add_argument('--sd', default='cifar10', type=str, help='source dataset')
parser.add_argument('--td', default='usps', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=165, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=30, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--layer', type=int, default=2, help='layer size : 14, 20, 32, 44, 56, 110')
parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')
parser.add_argument('--Method', default='BN', type=str, help='main Method BN|GN|P1|P2')
parser.add_argument('--group', type=int, default=1, help='number of group at GN')


source_prediction_max_result = []
target_prediction_max_result = []
best_prec_result = torch.tensor(0, dtype=torch.float32)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def main():
    global args, best_prec_result
    
    start_epoch = 0
    utils.default_model_dir = args.dir
    start_time = time.time()

    train_loader, test_loader, _, _ = dataset_selector(args.sd)

    state_info = utils.model_optim_state_info()
    state_info.model_init(args=args, num_class=10)
    state_info.model_cuda_init()
    # state_info.weight_init()
    state_info.optimizer_init(args)

    if cuda:
        print("USE", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True

    checkpoint = utils.load_checkpoint(utils.default_model_dir, is_last=False)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)

    check_selection(state_info, train_loader)
    check_selection(state_info, test_loader)

def check_selection(state_info, loader): # all 

    state_info.set_train_mode()
    for it, (x, y) in enumerate(loader):
        if x.size(0) is not args.batch_size:
            continue

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        output = state_info.forward(x)

    print('done')

def dataset_selector(data):
    if data == 'mnist':
        return dataset.MNIST_loader(img_size=args.img_size)
    elif data == 'svhn':
        return dataset.SVHN_loader(img_size=32)
    elif data == "usps":
        return dataset.usps_loader(img_size=args.img_size)
    elif data == "mnistm":
        return dataset.MNIST_M_loader(img_size=args.img_size)
    elif data == "cifar10":
        return dataset.cifar10_loader(args)

def to_var(x, dtype):
    return Variable(x.type(dtype))

if __name__=='__main__':
    main()