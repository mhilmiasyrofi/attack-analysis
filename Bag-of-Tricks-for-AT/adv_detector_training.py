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

import torchvision
from torchvision import datasets, transforms


from sklearn.utils import resample

import os

from wideresnet import WideResNet

## import the root project to the python environment
sys.path.insert(0,'/workspace/attack-analysis')
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
    parser.add_argument('--list', default='pixelattack_spatialtransformation_autopgd')
    parser.add_argument('--sample', default=100, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--val', default=-1, type=int)
    parser.add_argument('--adv-dir', default='../adv_examples/', type=str)
    parser.add_argument('--output-dir', default='../adv_detectors/', type=str)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
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
    
    names = ""
    
    if args.val != -1 :
        names += str(args.val) + "val/"
    else :
        names += "default/"
    
    if args.sample != 100 :
        names += str(args.sample) + "sample/" + args.list + "/"
    else :
        names += "full/" + args.list + "/"

    print('File name: ', names)
    return names


def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def main():
    args = get_args()

    output_dir = args.output_dir + get_auto_fname(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # setup data loader
    transformations = [Crop(32, 32), FlipLR()]
    if args.val != -1:
        np.random.seed(0)
        m = 50000
        P = np.random.permutation(m)
        n = args.val
        
        dataset = cifar10(args.data_dir)

        val_data = dataset['train']['data'][P[:n]]
        val_labels = [dataset['train']['labels'][p] for p in P[:n]]
        train_data = dataset['train']['data'][P[n:]]
        train_labels = [dataset['train']['labels'][p] for p in P[n:]]

        dataset['train']['data'] = train_data
        dataset['train']['labels'] = train_labels
        dataset['val'] = {
            'data' : val_data, 
            'labels' : val_labels
        }
        dataset['split'] = n
        dataset['permutation'] = P
        
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=4)
    else:
        dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    
    
    train_set = Transform(train_set, transformations)
    if args.sample != 100 :
        n = len(train_set) 
        n_sample = int(n * args.sample / 100)
        
        np.random.shuffle(train_set)
        train_set = train_set[:n_sample]

    train_batches = Batches(train_set, args.batch_size, shuffle=True, set_random_choices=True, num_workers=4)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    if args.val != -1:
        test_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=4)  
    else :
        test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=4)  
    
        

    print("")
    print("Train Original Data: ")
    print("Len: ", len(train_set))
    print("")
        
    shuffle = True
        
    train_adv_images = None
    train_adv_labels = None

    val_adv_images = None
    val_adv_labels = None
    
    test_adv_images = None
    test_adv_labels = None
    
    print("Attacks")
    attacks = args.list.split("_")
    print(attacks)
    
    for i in range(len(attacks)):
        _adv_dir = args.adv_dir + "{}/".format(attacks[i])
        train_path = _adv_dir + "train.pth" 
        test_path = _adv_dir + "test.pth"

        adv_train_data = torch.load(train_path)
        adv_test_data = torch.load(test_path)

        if i == 0 :
            train_adv_images = adv_train_data["adv"]
            train_adv_labels = [i] * len(adv_train_data["label"])
            if args.val != -1:
                permutation = dataset['permutation']
                split = dataset['split']
                val_adv_images = train_adv_images[permutation[:split]]
                train_adv_images = train_adv_images[permutation[split:]]
                val_adv_labels = [train_adv_labels[p] for p in permutation[:split]]
                train_adv_labels = [train_adv_labels[p] for p in permutation[split:]]
            test_adv_images = adv_test_data["adv"]
            test_adv_labels = [i] * len(adv_test_data["label"])
        else :
            curr_train_adv_images = adv_train_data["adv"]
            curr_train_adv_labels = [i] * len(adv_train_data["label"])
            if args.val != -1:
                permutation = dataset['permutation']
                split = dataset['split']
                curr_val_adv_images = curr_train_adv_images[permutation[:split]]
                curr_train_adv_images = curr_train_adv_images[permutation[split:]]
                curr_val_adv_labels = [curr_train_adv_labels[p] for p in permutation[:split]]
                curr_train_adv_labels = [curr_train_adv_labels[p] for p in permutation[split:]]
            curr_test_adv_images = adv_test_data["adv"]
            curr_test_adv_labels = [i] * len(adv_test_data["label"])
            
            train_adv_images = np.concatenate((train_adv_images, curr_train_adv_images))
            train_adv_labels = np.concatenate((train_adv_labels, curr_train_adv_labels))
            val_adv_images = np.concatenate((val_adv_images, curr_val_adv_images))
            val_adv_labels = np.concatenate((val_adv_labels, curr_val_adv_labels))  
            
            test_adv_images = np.concatenate((test_adv_images, curr_test_adv_images))
            test_adv_labels = np.concatenate((test_adv_labels, curr_test_adv_labels))  
        
    train_adv_set = list(zip(train_adv_images,
        train_adv_labels))

    if args.sample != 100 :
        n = len(train_adv_set) 
        n_sample = int(n * args.sample / 100)
        
        np.random.shuffle(train_adv_set)
        train_adv_set = train_adv_set[:n_sample]
        
    print("")
    print("Train Adv Attack Data: ", args.list)
    print("Len Train: ", len(train_adv_set))
    print("")

    train_adv_batches = Batches(train_adv_set, args.batch_size, shuffle=shuffle, set_random_choices=False, num_workers=4)
    
    
    if args.val != -1:
        test_adv_set = list(zip(val_adv_images, val_adv_labels))
    else :
        test_adv_set = list(zip(test_adv_images,
            test_adv_labels))
        
    test_adv_batches = Batches(test_adv_set, args.batch_size, shuffle=False, num_workers=4)
    print("Len Test: ", len(test_adv_set))
    
    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)

    # Set models
    model = None
    if args.model == "resnet18" :
        model = resnet18(num_classes=len(attacks))
    else:
        raise ValueError("Unknown model")

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

    best_train_adv_acc = 0
    best_test_adv_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(output_dir, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(output_dir, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_adv_acc = torch.load(os.path.join(output_dir, f'model_best.pth'))['test_adv_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")
        
    
    model.cuda()
    model.eval()
    
    # Evaluate on original test data
    test_acc = 0
    test_n = 0
    
    for i, batch in enumerate(test_batches):
        X, y = batch['input'], batch['target']

        clean_input = normalize(X)
        output = model(clean_input)
        
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)
        
    logger.info('Intial Accuracy on Original Test Data: %.4f (Test Acc)', test_acc/test_n)
    
    test_adv_acc = 0
    test_adv_n = 0
        
    for i, batch in enumerate(test_adv_batches):                            
        adv_input = normalize(batch['input'])
        y = batch['target']

        robust_output = model(adv_input)
        test_adv_acc += (robust_output.max(1)[1] == y).sum().item()
        test_adv_n += y.size(0)
    
    logger.info('Intial Accuracy on Adversarial Test Data: %.4f (Test Robust Acc)', test_adv_acc/test_adv_n)
        
    model.train()
    
    
    logger.info('Epoch \t Train Robust Acc \t Test Robust Acc')
    
    
    # Records per epoch for savetxt
    train_loss_record = []
    train_acc_record = []
    train_adv_loss_record = []
    train_adv_acc_record = []
    train_grad_record = []

    test_loss_record = []
    test_acc_record = []
    test_adv_loss_record = []
    test_adv_acc_record = []
    test_grad_record = []

    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()

        train_loss = 0
        train_acc = 0
        train_adv_loss = 0
        train_adv_acc = 0
        train_n = 0
        train_grad = 0

        record_iter = torch.tensor([])

        for i, adv_batch in enumerate(train_adv_batches):
            if args.eval:
                break
            adv_input = normalize(adv_batch['input'])
            adv_y = adv_batch['target']
            adv_input.requires_grad = True
            robust_output = model(adv_input)
            
            # Training losses
            if args.mixup:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            elif args.mixture:
                robust_loss = args.mixture_alpha * criterion(robust_output, adv_y) + (1-args.mixture_alpha) * criterion(output, y)
            else:
                if args.focalloss:
                    criterion_nonreduct = nn.CrossEntropyLoss(reduction='none')
                    robust_confidence = F.softmax(robust_output, dim=1)[:, adv_y].detach()
                    robust_loss = (criterion_nonreduct(robust_output, adv_y) * ((1. - robust_confidence) ** args.focallosslambda)).mean()

                elif args.use_DLRloss:
                    beta_ = 0.8 * epoch_now / args.epochs
                    robust_loss = (1. - beta_) * F.cross_entropy(robust_output, adv_y) + beta_ * dlr_loss(robust_output, adv_y)

                elif args.use_CWloss:
                    beta_ = 0.8 * epoch_now / args.epochs
                    robust_loss = (1. - beta_) * F.cross_entropy(robust_output, adv_y) + beta_ * CW_loss(robust_output, adv_y)

                elif args.use_FNandWN:
                    #print('use FN and WN with margin')
                    robust_loss = criterion(args.s_FN * robust_output - onehot_target_withmargin_HE, adv_y)

                else:
                    robust_loss = criterion(robust_output, adv_y)


            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            # Record the statstic values
            train_adv_loss += robust_loss.item() * adv_y.size(0)
            train_adv_acc += (robust_output.max(1)[1] == adv_y).sum().item()
            train_n += adv_y.size(0)

        train_time = time.time()
        if args.earlystopPGD:
            print('Iter mean: ', record_iter.mean().item(), ' Iter std:  ', record_iter.std().item())

        # Evaluate on test data
        model.eval()

        test_adv_loss = 0
        test_adv_acc = 0
        test_adv_n = 0
        
            
        for i, batch in enumerate(test_adv_batches):                            
            adv_input = normalize(batch['input'])
            y = batch['target']

            robust_output = model(adv_input)
            robust_loss = criterion(robust_output, y)

            test_adv_loss += robust_loss.item() * y.size(0)
            test_adv_acc += (robust_output.max(1)[1] == y).sum().item()
            test_adv_n += y.size(0)

        test_time = time.time()



        logger.info('%d \t %.4f \t\t %.4f',
            epoch+1, train_adv_acc/train_n, test_adv_acc/test_adv_n)

        # Save results
        train_adv_loss_record.append(train_adv_loss/train_n)
        train_adv_acc_record.append(train_adv_acc/train_n)

        np.savetxt(output_dir+'/train_adv_loss_record.txt', np.array(train_adv_loss_record))
        np.savetxt(output_dir+'/train_adv_acc_record.txt', np.array(train_adv_acc_record))

        test_adv_loss_record.append(test_adv_loss/train_n)
        test_adv_acc_record.append(test_adv_acc/train_n)

        np.savetxt(output_dir+'/test_adv_loss_record.txt', np.array(test_adv_loss_record))
        np.savetxt(output_dir+'/test_adv_acc_record.txt', np.array(test_adv_acc_record))

        # save checkpoint
        if epoch > 99 or (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(output_dir, f'opt_{epoch}.pth'))

        # save best
        if test_adv_acc/test_n > best_test_adv_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_adv_acc':test_adv_acc/test_n,
                    'test_adv_loss':test_adv_loss/test_n,
                }, os.path.join(output_dir, f'model_best.pth'))
            best_test_adv_acc = test_adv_acc/test_n
            best_train_adv_acc = train_adv_acc/train_n
        elif test_adv_acc/test_n == best_test_adv_acc and train_adv_acc/train_n > best_train_adv_acc :
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_adv_acc':test_adv_acc/test_n,
                    'test_adv_loss':test_adv_loss/test_n,
                }, os.path.join(output_dir, f'model_best.pth'))
            best_test_adv_acc = test_adv_acc/test_n
            best_train_adv_acc = train_adv_acc/train_n
            
if __name__ == "__main__":
    main()
