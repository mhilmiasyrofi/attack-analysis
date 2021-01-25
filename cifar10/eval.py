import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from wideresnet import WideResNet
from models import *

from utils import *

from constant import TOOLBOX_ADV_ATTACK_LIST

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
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--train-adversarial', default='autoattack')
    parser.add_argument('--list', default='newtonfool_pixelattack_spatialtransformation')
    parser.add_argument('--balanced', default=None) # "9_1_1"
    parser.add_argument('--test-adversarial', default='pgd')
    parser.add_argument('--best-model', action='store_true')
    parser.add_argument('--model-epoch', default=-1, type=int)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--data-dir', default='cifar-data', type=str)
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
#     parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--fname', default='../../trained_models/default/', type=str)
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
#     names = args.model + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
#     names = args.model + '_'  + args.train_adversarial + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
    
    names = None
    
#     if args.sample != 100 :
#         names = str(args.sample) + "/" + args.train_adversarial + "/"
#     else :
#         names = "default/" + args.train_adversarial + "/"

    
#     if args.train_adversarial == "combine" :
#         if args.balanced != None :            
#             names = args.model + '_'  + args.train_adversarial + '_balanced_' + args.list + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
#         else :
#             names = args.model + '_'  + args.train_adversarial + '_' + args.list + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
#     else :
#         names = args.model + '_'  + args.train_adversarial + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)    
    
#     # Group 1
#     if args.earlystopPGD:
#         names = names + '_earlystopPGD' + str(args.earlystopPGDepoch1) + str(args.earlystopPGDepoch2)
#     if args.warmup_lr:
#         names = names + '_warmuplr' + str(args.warmup_lr_epoch)
#     if args.warmup_eps:
#         names = names + '_warmupeps' + str(args.warmup_eps_epoch)
#     if args.weight_decay != 5e-4:
#         names = names + '_wd' + str(args.weight_decay)
#     if args.labelsmooth:
#         names = names + '_ls' + str(args.labelsmoothvalue)

#     # Group 2
#     if args.use_stronger_adv:
#         names = names + '_usestrongeradv#' + str(args.stronger_index)
#     if args.use_multitarget:
#         names = names + '_usemultitarget'
#     if args.use_DLRloss:
#         names = names + '_useDLRloss'
#     if args.use_CWloss:
#         names = names + '_useCWloss'
#     if args.use_FNandWN:
#         names = names + '_HE' + 's' + str(args.s_FN) + 'm' + str(args.m_FN)
#     if args.use_adaptive:
#         names = names + 'adaptive'
#     if args.use_FNonly:
#         names = names + '_FNonly'
#     if args.fast_better:
#         names = names + '_fastbetter'
#     if args.activation != 'ReLU':
#         names = names + '_' + args.activation
#         if args.activation == 'Softplus':
#             names = names + str(args.softplus_beta)
#     if args.lrdecay != 'base':
#         names = names + '_' + args.lrdecay
#     if args.BNeval:
#         names = names + '_BNeval'
#     if args.focalloss:
#         names = names + '_focalloss' + str(args.focallosslambda)
#     if args.optimizer != 'momentum':
#         names = names + '_' + args.optimizer
#     if args.mixup:
#         names = names + '_mixup' + str(args.mixup_alpha)
#     if args.cutout:
#         names = names + '_cutout' + str(args.cutout_len)

    print('File name: ', names)
    return names

def main():
    args = get_args()
#     names = get_auto_fname(args)
#     args.fname = args.fname + names

    args.fname += args.train_adversarial + "/"

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
        
#     eval_dir = args.fname + '/eval/' + args.test_adversarial + "/"

    eval_dir = args.fname + "eval/"
    if args.model_epoch != -1 :
        eval_dir += str(args.model_epoch) + "/"
    else :
        eval_dir += "best/"
    eval_dir += args.test_adversarial + "/"

    
    if not os.path.exists(eval_dir):
        print("Make dirs: ", eval_dir)
        os.makedirs(eval_dir)
    

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(eval_dir, "output.log")),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # Prepare data
    dataset = cifar10(args.data_dir)
    
    x_test = (dataset['test']['data']/255.)
    x_test = transpose(x_test).astype(np.float32)
    y_test = dataset['test']['labels']
    
    test_set = list(zip(x_test, y_test))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=4)
    
    print("Train Adv Attack Data: ", args.train_adversarial)
    
    adv_dir = "adv_examples/{}/".format(args.test_adversarial)
    train_path = adv_dir + "train.pth" 
    test_path = adv_dir + "test.pth"
    
    ATTACK_LIST = ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "newtonfool", "pgd", "pixelattack", "spatialtransformation", "squareattack"]
    
    if args.test_adversarial in TOOLBOX_ADV_ATTACK_LIST :
        adv_train_data = torch.load(train_path)
        test_adv_images_on_train = adv_train_data["adv"]
        test_adv_labels_on_train = adv_train_data["label"]
        adv_test_data = torch.load(test_path)
        test_adv_images_on_test = adv_test_data["adv"]
        test_adv_labels_on_test = adv_test_data["label"]
    elif args.test_adversarial == "all":
        for i in range(len(ATTACK_LIST)):
            _adv_dir = "adv_examples/{}/".format(ATTACK_LIST[i])
            train_path = _adv_dir + "train.pth" 
            test_path = _adv_dir + "test.pth"

            adv_train_data = torch.load(train_path)
            adv_test_data = torch.load(test_path)
            
            if i == 0 :
                train_adv_images = adv_train_data["adv"]
                train_adv_labels = adv_train_data["label"]
                test_adv_images = adv_test_data["adv"]
                test_adv_labels = adv_test_data["label"]   
            else :
#                 print(train_adv_images.shape)
#                 print(adv_train_data["adv"].shape)
                train_adv_images = np.concatenate((train_adv_images, adv_train_data["adv"]))
                train_adv_labels = np.concatenate((train_adv_labels, adv_train_data["label"]))
                test_adv_images = np.concatenate((test_adv_images, adv_test_data["adv"]))
                test_adv_labels = np.concatenate((test_adv_labels, adv_test_data["label"]))
                
            test_adv_images_on_train = train_adv_images
            test_adv_labels_on_train = train_adv_labels
            test_adv_images_on_test = test_adv_images
            test_adv_labels_on_test = test_adv_labels

    elif args.test_adversarial in ["ffgsm", "mifgsm", "tpgd"] :
        adv_data = {}
        adv_data["adv"], adv_data["label"] = torch.load(train_path)
        test_adv_images_on_train = adv_data["adv"].numpy()
        test_adv_labels_on_train = adv_data["label"].numpy()
        adv_data = {}
        adv_data["adv"], adv_data["label"] = torch.load(test_path)
        test_adv_images_on_test = adv_data["adv"].numpy()
        test_adv_labels_on_test = adv_data["label"].numpy()
    else :
        raise ValueError("Unknown adversarial data")
    
    print("Test Adv Attack Data: ", args.test_adversarial)
    
    test_adv_on_train_set = list(zip(test_adv_images_on_train,
        test_adv_labels_on_train))
    
    test_adv_on_train_batches = Batches(test_adv_on_train_set, args.batch_size, shuffle=False, set_random_choices=False, num_workers=4)
    
    test_adv_on_test_set = list(zip(test_adv_images_on_test,
        test_adv_labels_on_test))
        
    test_adv_on_test_batches = Batches(test_adv_on_test_set, args.batch_size, shuffle=False, num_workers=4)


    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)


    # Set models
    model = None
    if args.model == "resnet18" :
        model = resnet18(pretrained=True)
    elif args.model == "resnet20" :
        model = resnet20()
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

    model.cuda()
    model.train()

    # Set training hyperparameters
    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()
    if args.lr_schedule == 'cyclic':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        if args.optimizer == 'momentum':
            opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'Nesterov':
            opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'SGD_GC':
            opt = SGD_GC(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD_GCC':
            opt = SGD_GCC(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            opt = torch.optim.AdamW(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    
    # Cross-entropy (mean)
    if args.labelsmooth:
        criterion = LabelSmoothingLoss(smoothing=args.labelsmoothvalue)
    else:
        criterion = nn.CrossEntropyLoss()

    # If we use freeAT or fastAT with previous init
    if args.train_adversarial == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.train_adversarial == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.train_adversarial == 'free':
        epochs = int(math.ceil(args.epochs / args.train_adversarial_iters))
    else:
        epochs = args.epochs


    # Set lr schedule
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t, warm_up_lr = args.warmup_lr):
            if t < 100:
                if  warm_up_lr and t < args.warmup_lr_epoch:
                    return (t + 1.) / args.warmup_lr_epoch * args.lr_max
                else:
                    return args.lr_max
            if args.lrdecay == 'lineardecay':
                if t < 105:
                    return args.lr_max * 0.02 * (105 - t)
                else:
                    return 0.
            elif args.lrdecay == 'intenselr':
                if t < 102:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
            elif args.lrdecay == 'looselr':
                if t < 150:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
            elif args.lrdecay == 'base':
                if t < 105:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        def lr_schedule(t, stepsize=18, min_lr=1e-5, max_lr=args.lr_max):

            # Scaler: we can adapt this if we do not want the triangular CLR
            scaler = lambda x: 1.

            # Additional function to see where on the cycle we are
            cycle = math.floor(1 + t / (2 * stepsize))
            x = abs(t / stepsize - 2 * cycle + 1)
            relative = max(0, (1 - x)) * scaler(cycle)

            return min_lr + (max_lr - min_lr) * relative




    #### Set stronger adv attacks when decay the lr ####
    def eps_alpha_schedule(t, warm_up_eps = args.warmup_eps, if_use_stronger_adv=args.use_stronger_adv, stronger_index=args.stronger_index): # Schedule number 0
        if stronger_index == 0:
            epsilon_s = [epsilon * 1.5, epsilon * 2]
            pgd_alpha_s = [pgd_alpha, pgd_alpha]
        elif stronger_index == 1:
            epsilon_s = [epsilon * 1.5, epsilon * 2]
            pgd_alpha_s = [pgd_alpha * 1.25, pgd_alpha * 1.5]
        elif stronger_index == 2:
            epsilon_s = [epsilon * 2, epsilon * 2.5]
            pgd_alpha_s = [pgd_alpha * 1.5, pgd_alpha * 2]
        else:
            print('Undefined stronger index')

        if if_use_stronger_adv:
            if t < 100:
                if t < args.warmup_eps_epoch and warm_up_eps:
                    return (t + 1.) / args.warmup_eps_epoch * epsilon, pgd_alpha, args.restarts
                else:
                    return epsilon, pgd_alpha, args.restarts
            elif t < 105:
                return epsilon_s[0], pgd_alpha_s[0], args.restarts
            else:
                return epsilon_s[1], pgd_alpha_s[1], args.restarts
        else:
            if t < args.warmup_eps_epoch and warm_up_eps:
                return (t + 1.) / args.warmup_eps_epoch * epsilon, pgd_alpha, args.restarts
            else:
                return epsilon, pgd_alpha, args.restarts

    #### Set the counter for the early stop of PGD ####
    def early_stop_counter_schedule(t):
        if t < args.earlystopPGDepoch1:
            return 1
        elif t < args.earlystopPGDepoch2:
            return 2
        else:
            return 3

    best_test_adv_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_adv_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_adv_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0
        
    if args.model_epoch == -1:
        if args.train_adversarial == "original" :
            logger.info(f'Run using the original model')
        else :
            logger.info(f'Run using the best model')
            model.load_state_dict(torch.load(os.path.join(args.fname, f'model_best.pth'))["state_dict"])
    else :
        model.load_state_dict(torch.load(os.path.join(args.fname, 'model_' + str(args.model_epoch) + '.pth')))

    if args.eval:
        logger.info("[Evaluation mode]")

        
    # Evaluate on test data
    model.eval()
    
    test_loss = 0
    test_acc = 0
    test_n = 0
    y_original = np.array([])
    y_original_pred = np.array([])

    test_adv_test_loss = 0
    test_adv_test_acc = 0
    test_adv_test_n = 0
    y_adv = np.array([])
    y_adv_pred = np.array([])

    test_adv_train_loss = 0
    test_adv_train_acc = 0
    test_adv_train_n = 0
    


    for i, batch in enumerate(test_batches):
        X, y = batch['input'], batch['target']

        clean_input = normalize(X)
        output = model(clean_input)
        loss = criterion(output, y)

        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)
        
        y_original = np.append(y_original, y.cpu().numpy())
        y_original_pred = np.append(y_original_pred, output.max(1)[1].cpu().numpy())


    for i, batch in enumerate(test_adv_on_test_batches):
        adv_input = normalize(batch['input'])
        y = batch['target']

        cross_robust_output = model(adv_input)
        cross_robust_loss = criterion(cross_robust_output, y)

        test_adv_test_loss += cross_robust_loss.item() * y.size(0)
        test_adv_test_acc += (cross_robust_output.max(1)[1] == y).sum().item()
        test_adv_test_n += y.size(0)

        y_adv = np.append(y_adv, y.cpu().numpy())
        y_adv_pred = np.append(y_adv_pred, cross_robust_output.max(1)[1].cpu().numpy())

        
    for i, batch in enumerate(test_adv_on_train_batches):
        adv_input = normalize(batch['input'])
        y = batch['target']

        cross_robust_output = model(adv_input)
        cross_robust_loss = criterion(cross_robust_output, y)

        test_adv_train_loss += cross_robust_loss.item() * y.size(0)
        test_adv_train_acc += (cross_robust_output.max(1)[1] == y).sum().item()
        test_adv_train_n += y.size(0)

    test_time = time.time()

    logger.info("Test Acc \tTest Robust Acc on Test \tTest Robust Acc on Train")
    logger.info('%.4f \t\t %.4f \t\t %.4f', 
                test_acc/test_n,
                test_adv_test_acc/test_adv_test_n,
                test_adv_train_acc/test_adv_train_n)

    
    y_original = y_original.astype(np.int)
    y_original_pred = y_original_pred.astype(np.int)

    y_adv = y_adv.astype(np.int)
    y_adv_pred = y_adv_pred.astype(np.int)
    
    logger.info("Y_original")
    logger.info(y_original)
    np.savetxt(os.path.join(eval_dir, "Y_original.txt"), y_original,  fmt='%i')
    
    logger.info("Y_original_pred")
    logger.info(y_original_pred)
    np.savetxt(os.path.join(eval_dir, "Y_original_pred.txt"), y_original_pred, fmt='%i')
    
    logger.info("Y_adv")
    logger.info(y_adv)
    np.savetxt(os.path.join(eval_dir, "Y_adv.txt"), y_adv, fmt='%i')
    
    
    logger.info("Y_adv_pred")
    logger.info(y_adv_pred)
    np.savetxt(os.path.join(eval_dir, "Y_adv_pred.txt"), y_adv_pred, fmt='%i')
    
    
if __name__ == "__main__":
    main()
