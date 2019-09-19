#python main.py -a pdvgg16_bn --data_train /shared/imagenet/train --data_val /shared/xli2/val --batch-size 64 --workers 4 --gpu 7 --ckptdirprefix experiment_1/ --epochs 1
#######################################################################################################################
#
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
# ********************************************************************************************************************
#
#
# The code in this file is adapted from: https://github.com/pytorch/examples/tree/master/imagenet/main.py
#
# Main Difference from the original file: add the networks using partial convolution based padding
#
# Network options using zero padding:               vgg16_bn, vgg19_bn, resnet50, resnet101, resnet152, ... 
# Network options using partial conv based padding: pdvgg16_bn, pdvgg19_bn, pdresnet50, pdresnet101, pdresnet152, ...
#
# Contact: Guilin Liu (guilinl@nvidia.com)
#
#######################################################################################################################
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.models as models_baseline # networks with zero padding
import models as models_partial # partial conv based padding 


model_baseline_names = sorted(name for name in models_baseline.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_baseline.__dict__[name])) # get a list of names of neural network models

model_partial_names = sorted(name for name in models_partial.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_partial.__dict__[name]))# get another list of names of neural network models

model_names = model_baseline_names + model_partial_names


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset') # create an object to aid in parsing of passed commandline arguments
parser.add_argument('--data_train', metavar='DIRTRAIN',
                    help='path to training dataset')

parser.add_argument('--data_val', metavar='DIRVAL',
                    help='path to validation dataset')                    

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# use the batch size 256 or 192 depending on the memeory
parser.add_argument('-b', '--batch-size', default=192, type=int,
                    metavar='N', help='mini-batch size (default: 192)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--ckptdirprefix', default='', type=str)

best_prec1 = 0


def main():
    global args, best_prec1 #global keyword allows local references to global variables
    args = parser.parse_args() #parse the arguments

    checkpoint_dir = args.ckptdirprefix + 'checkpoint_' + args.arch + '_' + args.prefix + '/' #directory of checkpoint
    if not os.path.exists(checkpoint_dir): #check path
        os.makedirs(checkpoint_dir) #make path
    args.logger_fname = os.path.join(checkpoint_dir, 'loss.txt') # create path and file name

    with open(args.logger_fname, "a") as log_file: #open a file for appending
        now = time.strftime("%c")#get the time formatted in locale appropriate representation
        log_file.write('================ Training Loss (%s) ================\n' % now)    #begin logging with timestamp
        log_file.write('world size: %d\n' % args.world_size)# print the world size (number of distributed processes
		
		
    if args.seed is not None:
        random.seed(args.seed)#seed python's random number generator
        torch.manual_seed(args.seed)#seed pytorch's random number generator
        cudnn.deterministic = True #enforce determinism in pytorch for a given distribution and hardware
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.') #warn about performance penalty of determinism

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')#warn about loss of data parallelism when a specific gpu is selected

    args.distributed = args.world_size > 1 #set whether or not distributed

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)#initiate the pytorch distributed process group with the specified backend, init_method, and world size#init method is an ip address to look for other process, world size is similar to number of processes

    # create model
    if args.pretrained: #if using a pretrained model
        print("=> using pre-trained model '{}'".format(args.arch)) #using a specific architecture's pre-trained model
        if args.arch in models_baseline.__dict__:#if args.arch in models_baseline, set the model
            model = models_baseline.__dict__[args.arch](pretrained=True)#set the model
        else:#arch was incorrectly set
            model = models_partial.__dict__[args.arch](pretrained=True)#use models partial??
        # model = models.__dict__[args.arch](pretrained=True)
    else: #don't use a pretrained model
        print("=> creating model '{}'".format(args.arch))#create model
        if args.arch in models_baseline.__dict__: #check baseline dict
            model = models_baseline.__dict__[args.arch]()#assign model
        else:
            model = models_partial.__dict__[args.arch]()#assign model from models_partial
        # model = models.__dict__[args.arch]()


    # logging
    with open(args.logger_fname, "a") as log_file: #append log file
        log_file.write('model created\n') # write to log
		
		
    if args.gpu is not None:#if argsgpu is not none
        model = model.cuda(args.gpu)#set gpu to run on
    elif args.distributed: #
        model.cuda() #initialize gpu, copy data from cpu
        model = torch.nn.parallel.DistributedDataParallel(model) #prep model to run with parallel, distributed data ?
    else:
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        if args.arch.startswith('alexnet') or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features) #modify the model features to be parallel and distributed
            model.cuda()#initialize cuda gpus
        else:
            model = torch.nn.DataParallel(model).cuda() #make the model parallel and move to gpu

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu) # create a loss function?

    # [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)#build an optimizer for the loss function using stochastic gradient descent

    # optionally resume from a checkpoint
    if args.resume: #if resuming from a checkpoint
        if os.path.isfile(args.resume): #check path
            print("=> loading checkpoint '{}'".format(args.resume)) #loading checkpoint if path exists
            checkpoint = torch.load(args.resume) #load model from path
            args.start_epoch = checkpoint['epoch'] #set the epoch
            best_prec1 = checkpoint['best_prec1']#set the best precl??

            model.load_state_dict(checkpoint['state_dict'])#load deserialized object state dictionary
            
            optimizer.load_state_dict(checkpoint['optimizer'])#load deserialized optimizer
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    traindir = args.data_train #os.path.join(args.data, 'train') initialize local variable
    valdir = args.data_val  #os.path.join(args.data, 'val') initialize local variable
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], #image = (image -mean) / std  for each channel
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224), #data augmentation, crop an image to a given size and default scale
            transforms.RandomHorizontalFlip(), #flip images horizontally
            transforms.ToTensor(), #convert image to numpy array
            normalize, #apply the normalization transform
        ]))

    if args.distributed: #if distributed
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) #load subset of dataset
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(#create a data loader with the following values
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, #batch_size set to argument batch size
        num_workers=args.workers, pin_memory=True)  #number of worker threads set to argument number of workers, pin memomory is true, which speeds up gpu training by allocating memory for swap space between cpu and gpu

    # logging
    with open(args.logger_fname, "a") as log_file: #perform logging
        log_file.write('training/val dataset created\n') #write the log_file


    if args.evaluate: #if in evaluation mode
        validate(val_loader, model, criterion) #call validate instead of train and return
        return #return instead of continuing program


    # logging
    with open(args.logger_fname, "a") as log_file: ##not in evaluate mode, training
        log_file.write('started training\n') #write to log file that you are training

    for epoch in range(args.start_epoch, args.epochs): #iterate through the epochs
        if args.distributed: # if distributed
            train_sampler.set_epoch(epoch) # set the number of epochs in train sampler
        adjust_learning_rate(optimizer, epoch) #change the learning rate"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch) #train model for one epoch using the specified loss function and optimizer

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, foldername=checkpoint_dir, filename='checkpoint.pth.tar')


        if epoch >= 94:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False, foldername=checkpoint_dir, filename='epoch_'+str(epoch)+'_checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter() #create batchtime
    data_time = AverageMeter() #create datatime
    losses = AverageMeter() #create losses
    top1 = AverageMeter() #create top1
    top5 = AverageMeter() #create top5

    # switch to train mode
    model.train() #switch to training mode

    end = time.time() #set end to now
    for i, (input, target) in enumerate(train_loader): #enumerate and iterate through the train loader training set values
        # measure data loading time
        data_time.update(time.time() - end)#update date time by delta between when end was created and current time on cpu

        if args.gpu is not None: #if set to specific gpu
            input = input.cuda(args.gpu, non_blocking=True) #set to specific gpu and do not block other processes
        target = target.cuda(args.gpu, non_blocking=True) # set target to specific gpu and do not block other processes

        # compute output
        output = model(input)#compute the output
        loss = criterion(output, target)#calculate the loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5)) #how many does it get right ??
        losses.update(loss.item(), input.size(0)) #update the losses for the batch
        top1.update(prec1[0], input.size(0))#update the accuracy for top1
        top5.update(prec5[0], input.size(0))#update the accuracy for top5

        # compute gradient and do SGD step
        optimizer.zero_grad() #zero gradients between training examples
        loss.backward()#loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x.x.grad += dloss/dx
        optimizer.step()#optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs: #e.g. x += -lr * x.grad
        #https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
        # measure elapsed time
        batch_time.update(time.time() - end) #update the batch time to reflect difference between current time and when end was initialized
        end = time.time() #update end to current time

        if i % args.print_freq == 0: #if time to print
            print('Epoch: [{0}][{1}/{2}]\t' #print to three digits of precision
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            with open(args.logger_fname, "a") as log_file: #append to log file
                log_file.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter() #create batchtime
    losses = AverageMeter() #create losses
    top1 = AverageMeter() #create top1
    top5 = AverageMeter() #create top5

    # switch to evaluate mode
    model.eval() #switch to evaluation mode

    with torch.no_grad(): #set requires grad flags to false
        end = time.time() #update end to now
        for i, (input, target) in enumerate(val_loader): #enumerate and iterate through validation examples
            if args.gpu is not None: #if specific gpu assisgned
                input = input.cuda(args.gpu, non_blocking=True) #assign input to specific gpu and do not block other processes
            target = target.cuda(args.gpu, non_blocking=True) #assign input to specific gpu and do not block other processes

            # compute output
            output = model(input)#output is equal to the prediction the model makes given the input
            loss = criterion(output, target) #calculate the loss from output vs target

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5)) #obtain the accuracy
            losses.update(loss.item(), input.size(0)) #update the losses
            top1.update(prec1[0], input.size(0)) #update top1
            top5.update(prec5[0], input.size(0)) #update top5

            # measure elapsed time
            batch_time.update(time.time() - end) #update the duration for the batches
            end = time.time() #update end to now

            if i % args.print_freq == 0:#print
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

                with open(args.logger_fname, "a") as log_file: #append to log file
                    log_file.write('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5)) #print to log file

        with open(args.logger_fname, "a") as final_log_file:
            final_log_file.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, foldername='', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(foldername, filename))#save the state of the model at the given path
    if is_best:#copy if best
        shutil.copyfile(os.path.join(foldername, filename), os.path.join(foldername, 'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):#tensor, tensor, tupple
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()#TRANSPOSE A TENSOR
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #equality tensor returned--comparing target and pred after being converted to equivalent sizes

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) #sum up how many are correct
            res.append(correct_k.mul_(100.0 / batch_size)) #append the number of correct to the result
        return res #return the result


if __name__ == '__main__':
    main()
