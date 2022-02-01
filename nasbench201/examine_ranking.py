import os
import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.stats as ss

from nasbench201.model import Network, distill
from nas_201_api import NASBench201API as API
import optimizers.darts.utils as utils

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='/remote-home/share/share/dataset',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--save', type=str, default='darts_0', help='choose which supernet to load')
args = parser.parse_args()


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input, updateType='weight')
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
    return top1.avg, objs.avg


def load_checkpoint(model, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def get_accuracies(model, api, valid_queue, criterion, num_archs=100):
    acc_oneshots, acc_trues = [], []
    for i in range(num_archs):
        if not i == 0:
            arch = torch.randn_like(model._arch_parameters)
            model._arch_parameters.data = arch
        model.binarization()
        acc_oneshot, _ = infer(valid_queue, model, criterion)
        acc_oneshots.append(acc_oneshot.item())

        result = api.query_by_arch(model.genotype(), hp='200')
        cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
            cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
        acc_trues.append(cifar10_test)
    return acc_oneshots, acc_trues


def get_kendalltau(acc_oneshots, acc_trues):
    rank_onehots = ss.rankdata(acc_oneshots, method='min')
    rank_trues = ss.rankdata(acc_trues, method='min')

    tau, p_value = ss.kendalltau(rank_onehots, rank_trues)
    return tau


def main():
    torch.set_num_threads(3)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    if args.save == 'darts_0':
        # save = 'search-baseline-20200223-215626-0-6930'
        save = 'search-exp-20210916-120838-2-4022'
    elif args.save == 'darts_1':
        save = 'search-baseline-20200223-215626-1-859'
    elif args.save == 'random_0':
        save = 'search-baseline-20200223-215626-0-alpha-random-0.3-2195'
    elif args.save == 'random_1':
        save = 'search-baseline-20200223-215626-1-alpha-random-0.3-529'
    elif args.save == 'pgd_0':
        save = 'search-baseline-20200223-215626-0-alpha-pgd_linf-0.3-6660'
    elif args.save == 'pgd_1':
        save = 'search-baseline-20200223-215626-1-alpha-pgd_linf-0.3-5746'

    save = os.path.join(
        '/yepeng/exp2/1RobustDarts/SmoothDARTS/experiments/nasbench201/cifar10', save)
    
    api = API('/remote-home/share/share/dataset/NAS-Bench-201-v1_0-e61699.pth')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(C=16, N=5, max_nodes=4, num_classes=10, criterion=criterion).cuda()
    model = load_checkpoint(model, save)

    acc_oneshots, acc_trues= get_accuracies(model, api, valid_queue, criterion, 100)
    tau = get_kendalltau(acc_oneshots, acc_trues)


    print(acc_oneshots[:100])
    print(acc_trues[:100])
    print(tau)


if __name__ == '__main__':
    main()


















