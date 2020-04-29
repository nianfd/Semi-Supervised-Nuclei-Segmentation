# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
from sklearn.externals import joblib



arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

global_step = 0
outfile = open('models/dsb2018_96_NestedUNet_woDS/trainloss.txt', 'w')
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # cudnn.enabled = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    #print('alpha is : ' + str(alpha))
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def train(args, label_train_loader, unlabel_train_loader, model, ema_model, criterion, criterion_2, optimizer, epoch, scheduler=None):
    global global_step
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()
    ema_model.train()

    
    for j, (unlabelinput, _) in tqdm(enumerate(unlabel_train_loader), total=len(unlabel_train_loader)):
        unlabelinput = unlabelinput.cuda()
        for i, (input, target) in tqdm(enumerate(label_train_loader), total=len(label_train_loader)):
            torch.cuda.empty_cache()
            input = input.cuda()
            target = target.cuda()
            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                labeloutput_2 = ema_model(input)
                unlabeloutput_1 = model(unlabelinput)
                unlabeloutput_2 = ema_model(unlabelinput)
                
                loss1 = criterion(output, target)
                loss2 = criterion_2(unlabeloutput_1, unlabeloutput_2)
                loss3 = criterion_2(output, labeloutput_2)
                loss = loss1 + loss2 + loss3
                iou = iou_score(output, target)
            runningloss = 0    
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            runningloss += loss.item()
            outfile.write((str(runningloss))+'\n')
            outfile.flush()
            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            update_ema_variables(model, ema_model, 0.999, global_step)
        
        

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            torch.cuda.empty_cache()
            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()    
    criterion_2 = nn.MSELoss().cuda()
    #cudnn.benchmark = True
    # Data loading code
    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_val_img_paths, test_img_paths, train_val_mask_paths, testmask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    #Semi-Supervised
    fulldatasize = len(train_val_img_paths) + len(test_img_paths)
    valdatasize = int(fulldatasize*0.20)
    #print(valdatasize)
    val_img_paths = train_val_img_paths[:valdatasize]
    val_mask_paths= train_val_mask_paths[:valdatasize]
    
    traindatasize = len(train_val_img_paths) - valdatasize
    
    label_train_img_paths = train_val_img_paths[valdatasize + int(traindatasize * 0.95):len(train_val_img_paths)]
    label_train_mask_paths= train_val_mask_paths[valdatasize + int(traindatasize * 0.95):len(train_val_img_paths)]

    unlabel_img_paths = train_val_img_paths[valdatasize : int(traindatasize * 0.95)]
    unlabel_mask_paths = train_val_mask_paths[valdatasize : int(traindatasize * 0.95)]
    
    label_train_dataset = Dataset(args, label_train_img_paths, label_train_mask_paths, args.aug)
    unlabel_train_dataset = Dataset(args, unlabel_img_paths, unlabel_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    
    print(len(label_train_img_paths))
    #print(val_mask_paths)

    label_train_loader = torch.utils.data.DataLoader(
        label_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False, num_workers=0)
    
    unlabel_train_loader = torch.utils.data.DataLoader(
        unlabel_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False, num_workers=0)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    
    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)
    ema_model = archs.__dict__[args.arch](args)

    model = model.cuda()
    ema_model = model.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)



    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, label_train_loader, unlabel_train_loader, model, ema_model, criterion, criterion_2, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    set_seed(41)
    torch.cuda.empty_cache()
    main()
