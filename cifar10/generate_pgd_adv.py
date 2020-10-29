import argparse
import logging
import sys
import time
import math

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel

import os

from wideresnet import WideResNet
from models import *

from utils import *

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Batches():
    def __init__(self, dataset, batch_size, shuffle, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] 
        * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return loss_value.mean()

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()    

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, mixup=False, y_a=None, y_b=None, lam=None, 
               early_stop=False, early_stop_pgd_max=1,
               multitarget=False,
               use_DLRloss=False, use_CWloss=False,
               epoch=0, totalepoch=110, gamma=0.8,
               use_adaptive=False, s_HE=15,
               fast_better=False, BNeval=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # early stop pgd counter for each x
        early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).cuda()

        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            output = model(normalize(X + delta))

            # if use early stop pgd
            if early_stop:
                # calculate mask for early stop pgd
                if_success_fool = (output.max(1)[1] != y).to(dtype=torch.int32)
                early_stop_pgd_count = early_stop_pgd_count - if_success_fool
                index = torch.where(early_stop_pgd_count > 0)[0]
                iter_count[index] = iter_count[index] + 1
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            # Whether use mixup criterion
            if fast_better:
                loss_ori = F.cross_entropy(output, y)
                grad_ori = torch.autograd.grad(loss_ori, delta, create_graph=True)[0]
                loss_grad = (alpha / 4.) * (torch.norm(grad_ori.view(grad_ori.shape[0], -1), p=2, dim=1) ** 2)
                loss = loss_ori + loss_grad.mean()
                loss.backward()
                grad = delta.grad.detach()

            elif not mixup:
                if multitarget:
                    random_label = torch.randint(low=0, high=10, size=y.shape).cuda()
                    random_direction = 2*((random_label == y).to(dtype=torch.float32) - 0.5)
                    loss = torch.mean(random_direction * F.cross_entropy(output, random_label, reduction='none'))
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_DLRloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * dlr_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_CWloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * CW_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                else:
                    if use_adaptive:
                        loss = F.cross_entropy(s_HE * output, y)
                    else:
                        loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
            else:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
                loss.backward()
                grad = delta.grad.detach()


            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if BNeval:
        model.train()

    return max_delta, iter_count



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--test-pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=100, type=int)
    parser.add_argument('--mixture', action='store_true') # whether use mixture of clean and adv examples in a mini-batch
    parser.add_argument('--mixture_alpha', type=float)
    parser.add_argument('--l2', default=0, type=float)

    # Group 1
    parser.add_argument('--earlystopPGD', action='store_true') # whether use early stop in PGD
    parser.add_argument('--earlystopPGDepoch1', default=60, type=int)
    parser.add_argument('--earlystopPGDepoch2', default=100, type=int)

    parser.add_argument('--warmup_lr', action='store_true') # whether warm_up lr from 0 to max_lr in the first n epochs
    parser.add_argument('--warmup_lr_epoch', default=15, type=int)

    parser.add_argument('--weight_decay', default=5e-4, type=float)#weight decay

    parser.add_argument('--warmup_eps', action='store_true') # whether warm_up eps from 0 to 8/255 in the first n epochs
    parser.add_argument('--warmup_eps_epoch', default=15, type=int)

    parser.add_argument('--batch-size', default=128, type=int) #batch size

    parser.add_argument('--labelsmooth', action='store_true') # whether use label smoothing
    parser.add_argument('--labelsmoothvalue', default=0.0, type=float)

    parser.add_argument('--lrdecay', default='base', type=str, choices=['intenselr', 'base', 'looselr', 'lineardecay'])

    # Group 2
    parser.add_argument('--use_DLRloss', action='store_true') # whether use DLRloss
    parser.add_argument('--use_CWloss', action='store_true') # whether use CWloss


    parser.add_argument('--use_multitarget', action='store_true') # whether use multitarget

    parser.add_argument('--use_stronger_adv', action='store_true') # whether use mixture of clean and adv examples in a mini-batch
    parser.add_argument('--stronger_index', default=0, type=int)

    parser.add_argument('--use_FNandWN', action='store_true') # whether use FN and WN
    parser.add_argument('--use_adaptive', action='store_true') # whether use s in attack during training
    parser.add_argument('--s_FN', default=15, type=float) # s in FN
    parser.add_argument('--m_FN', default=0.2, type=float) # s in FN

    parser.add_argument('--use_FNonly', action='store_true') # whether use FN only

    parser.add_argument('--fast_better', action='store_true')

    parser.add_argument('--BNeval', action='store_true') # whether use eval mode for BN when crafting adversarial examples

    parser.add_argument('--focalloss', action='store_true') # whether use focalloss
    parser.add_argument('--focallosslambda', default=2., type=float)

    parser.add_argument('--activation', default='ReLU', type=str)
    parser.add_argument('--softplus_beta', default=1., type=float)

    parser.add_argument('--optimizer', default='momentum', choices=['momentum', 'Nesterov', 'SGD_GC', 'SGD_GCC', 'Adam', 'AdamW'])

    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)

    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)

    return parser.parse_args()

def get_auto_fname(args):
    names = args.model + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
    # Group 1
    if args.earlystopPGD:
        names = names + '_earlystopPGD' + str(args.earlystopPGDepoch1) + str(args.earlystopPGDepoch2)
    if args.warmup_lr:
        names = names + '_warmuplr' + str(args.warmup_lr_epoch)
    if args.warmup_eps:
        names = names + '_warmupeps' + str(args.warmup_eps_epoch)
    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)
    if args.labelsmooth:
        names = names + '_ls' + str(args.labelsmoothvalue)

    # Group 2
    if args.use_stronger_adv:
        names = names + '_usestrongeradv#' + str(args.stronger_index)
    if args.use_multitarget:
        names = names + '_usemultitarget'
    if args.use_DLRloss:
        names = names + '_useDLRloss'
    if args.use_CWloss:
        names = names + '_useCWloss'
    if args.use_FNandWN:
        names = names + '_HE' + 's' + str(args.s_FN) + 'm' + str(args.m_FN)
    if args.use_adaptive:
        names = names + 'adaptive'
    if args.use_FNonly:
        names = names + '_FNonly'
    if args.fast_better:
        names = names + '_fastbetter'
    if args.activation != 'ReLU':
        names = names + '_' + args.activation
        if args.activation == 'Softplus':
            names = names + str(args.softplus_beta)
    if args.lrdecay != 'base':
        names = names + '_' + args.lrdecay
    if args.BNeval:
        names = names + '_BNeval'
    if args.focalloss:
        names = names + '_focalloss' + str(args.focallosslambda)
    if args.optimizer != 'momentum':
        names = names + '_' + args.optimizer
    if args.mixup:
        names = names + '_mixup' + str(args.mixup_alpha)
    if args.cutout:
        names = names + '_cutout' + str(args.cutout_len)
    if args.attack != 'pgd':
        names = names + '_' + args.attack

    print('File name: ', names)
    return names


def main():
    args = get_args()
    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = 'trained_models/' + names
    else:
        args.fname = 'trained_models/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare data
    
    dataset = cifar10(args.data_dir)
        
    test_set = list(zip(transpose(dataset['train']['data']/255.),
        dataset['train']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=True, num_workers=4)
    

    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)

    # Set models
    model = None
    if args.model == "resnet18" :
        model = resnet18(pretrained=True)
    elif args.model == "vgg16bn" :
        model = vgg16_bn(pretrained=True)
    elif args.model == "densenet121" :
        model = densenet121(pretrained=True)
    elif args.model == "googlenet" :
        model = googlenet(pretrained=True)
    elif args.model == "inceptionv3" :
        model = inception_v3(pretrained=True)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0, normalize=args.use_FNandWN,
            activation=args.activation, softplus_beta=args.softplus_beta)
    elif args.model == 'WideResNet_20':
        model = WideResNet(34, 10, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN,
            activation=args.activation, softplus_beta=args.softplus_beta)
    else:
        raise ValueError("Unknown model")

    # Cross-entropy (mean)
    if args.labelsmooth:
        criterion = LabelSmoothingLoss(smoothing=args.labelsmoothvalue)
    else:
        criterion = nn.CrossEntropyLoss()
        
        
    model.cuda()
    model.eval()
    
    # Evaluate on test data
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    test_grad = 0

    originals = None
    adversarials = None
    labels = None

    for i, batch in enumerate(test_batches):
        X, y = batch['input'], batch['target']

        # Random initialization
        if args.attack == 'none':
            delta = torch.zeros_like(X)
        else:
            delta, _ = attack_pgd(model, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=False)
        delta = delta.detach()

        adv_input = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
        normalized_adv_input = normalize(adv_input)
        normalized_adv_input.requires_grad = True
        
        robust_output = model(normalized_adv_input)
        robust_loss = criterion(robust_output, y)

        clean_input = normalize(X)
        clean_input.requires_grad = True     
        output = model(clean_input)
        loss = criterion(output, y)

        # Get the gradient norm values
        input_grads = torch.autograd.grad(loss, clean_input, create_graph=False)[0]

        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)
        test_grad += input_grads.abs().sum()
        
        if i == 0 :
            originals = X.cpu().detach().numpy()
            adversarials = adv_input.cpu().detach().numpy()
            labels = y.cpu().detach().numpy()
        else :
            originals = np.concatenate((originals, X.cpu().detach().numpy()), axis=0)
            adversarials  = np.concatenate((adversarials, adv_input.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, y.cpu().detach().numpy()),axis=0)

    test_time = time.time()

    torch.save({"ori": torch.from_numpy(np.array(originals)), "adv": torch.from_numpy(np.array(adversarials)), "label": torch.from_numpy(np.array(labels))}, "data/pgd_adversarial_data.pt")
    
    logger.info('Test Acc: %.4f \t Test Robust Acc: %.4f',
                test_acc/test_n, test_robust_acc/test_n)

if __name__ == "__main__":
    main()
