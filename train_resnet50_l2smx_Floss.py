import argparse
import os
import visdom
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from face_dataset import MSCeleb
from tqdm import tqdm
import face_models
from  new_loss import FocalLoss as FL

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/scratch1', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='l2resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=15, type=int,
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
parser.add_argument('-m', '--model_path', default='/scratch1/bylu', type=str,
                    help='dir to save model')


def main():
    global args
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    viz = visdom.Visdom()
    loss_plt = viz.line(X=np.zeros(1),
                        Y=np.zeros(1),
                        opts=dict(markers=False))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = face_models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = face_models.__dict__[args.arch](num_classes=58207)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    # criterion = FL() # using focal loss

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[128. / 255, 128. / 255, 128. / 255])

    train_dataset = MSCeleb('/', traindir, transforms.Compose(
        [transforms.Scale((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize,
         ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, loss_plt)

        # save model
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.model_path, args.arch)


def train(train_loader, model, optimizer, epoch, loss_plt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_r = 0
    viz = visdom.Visdom()
    # switch to train mode
    model.train()

    end = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in pbar:
        # data_time.update(time.time() - end)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        output, _ = model(input_var)

        (N, C) = output.size()
        target = torch.zeros(N, C).scatter_(1, target.view(-1, 1), 1)
        target = target.cuda()
        target_var = torch.autograd.Variable(target)

        loss = FL(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        l_r = optimizer.param_groups[0]['lr']

        pbar.set_description('Epoch %i' % epoch)
        pbar.set_postfix({'iter/total':str(i)+'/'+str(len(train_loader)), 'loss':"%.3f" % losses.val, 'time/batch':"%.2f" % batch_time.val, 'lr':l_r})
        # print len(train_loader)*epoch+i
        # print '\n'
        # print losses.val
        viz.updateTrace(X=np.array([len(train_loader)*epoch+i]),
                        Y=np.array([losses.val]),
                        win=loss_plt)
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Train time/batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data load time/batch {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'
        #         .format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, path, arch,  snapshot='snapshot.pth.tar'):
    filename = os.path.join(path, arch + '_FL_' + str(state['epoch']) + '_' + snapshot)
    print("=> saving snapshot '{}' (epoch {})"
          .format(filename, state['epoch']))
    torch.save(state, filename)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.maximum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.maximum = max(self.maximum, val)


def adjust_learning_rate(optimizer, epoch, MultiEpoch=5, alpha=0.1):
    """
    Sets the learning rate to the initial LR decayed by 10 every 5 epochs

    :param optimizer: change optimizer directly
    :param epoch: current epoch
    :param multiepoch: how many epoch to reduce lr
    :param alpha: lr change rate
    :return:
    """
    lr = args.lr * (alpha ** (epoch // MultiEpoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
