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
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=165, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=30, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--layer', type=int, default=2, help='layer size : 14, 20, 32, 44, 56, 110')
parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')
parser.add_argument('--Method', default='BN', type=str, help='main Method BN|GN|P1|P2')
parser.add_argument('--Mode', type=int, default=0, help='0: first, 1: last, 2: all, 3: None')
parser.add_argument('--norm-type', type=str, default='batch', help='batch or instance')

source_prediction_max_result = []
target_prediction_max_result = []
best_prec_result = torch.tensor(0, dtype=torch.float32)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

criterion = torch.nn.CrossEntropyLoss()

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

    checkpoint = utils.load_checkpoint(utils.default_model_dir, is_last=True)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)

    for epoch in range(0, args.epoch):
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        for param_group in state_info.optimizer.param_groups:
            param_group['lr'] = lr

        train(state_info, train_loader, epoch)
        prec_result = test(state_info, test_loader, epoch)

        if prec_result > best_prec_result:
            best_prec_result = prec_result
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)
            utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))
    
    print('done')


def train(state_info, train_loader, epoch): # all 

    utils.print_log('Type, Epoch, Batch, loss, Percent')
    state_info.set_train_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)
    train_loss = 0

    for it, (x, y) in enumerate(train_loader):
        if x.size(0) is not args.batch_size:
            continue

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        output = state_info.forward(x)
        
        #  Train 
        state_info.optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        state_info.optimizer.step()
        
        #  Log Print
        train_loss += loss.data.item()
        total += float(y.size(0))
        _, predicted = torch.max(output.data, 1)
        correct += float(predicted.eq(y.data).cpu().sum())

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.4f}, {:.2f}'
                  .format(epoch, it, loss.item(), train_loss, 100.*correct / total))
            print('Train, {}, {}, {:.6f}, {:.4f}, {:.2f}'
                  .format(epoch, it, loss.item(), train_loss, 100.*correct / total))

    utils.print_log('')

def test(state_info, test_loader, epoch):

    utils.print_log('Type, Epoch, Batch, Acc')
    state_info.set_test_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(test_loader):
        if x.size(0) is not args.batch_size:
            continue

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        output = state_info.forward(x)

        #  Log Print
        total += float(y.size(0))
        _, predicted = torch.max(output.data, 1)
        correct += float(predicted.eq(y.data).cpu().sum())
        
    utils.print_log('Test, {}, {:.2f}'.format(epoch, 100.*correct / total))
    print('Test, {}, {:.2f}'.format(epoch, 100.*correct / total))
    
    utils.print_log('')
    return 100.*correct / total


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
    elif data == "cifar100":
        return dataset.cifar100_loader(args)

def to_var(x, dtype):
    return Variable(x.type(dtype))

if __name__=='__main__':
    main()