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
from torchvision import datasets, transforms

## import the root project to the python environment
sys.path.insert(0,'/workspace/attack-analysis')
from models import *

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18

from utils import *

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--attack', default='pgd')
    parser.add_argument('--sample', default=100, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--val', default=-1, type=int)
    parser.add_argument('--adv-dir', default='../adv_examples/', type=str)
    parser.add_argument('--output-dir', default='../trained_models/AT/', type=str)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=20, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    
    dirname = args.output_dir + "/"
        
    if args.val != -1 :
        dirname += str(args.val) + "val/"
    else :
        dirname += "default/"

    if args.sample == 100 :
        dirname += "full/" + args.attack + "/"
    else :
        dirname += str(args.sample) + "/" + args.attack + "/"

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(dirname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

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
        
    shuffle = False
        
    train_adv_images = None
    train_adv_labels = None

    val_adv_images = None
    val_adv_labels = None
    
    test_adv_images = None
    test_adv_labels = None

    adv_dir = args.adv_dir + "{}/".format(args.attack)
    train_path = adv_dir + "train.pth" 
    test_path = adv_dir + "test.pth"
    
    ATTACK_LIST = ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "newtonfool", "pgd", "pixelattack", "spatialtransformation", "squareattack"]
    

    if args.attack in ATTACK_LIST :
        adv_train_data = torch.load(train_path)
        train_adv_images = adv_train_data["adv"]
        train_adv_labels = adv_train_data["label"]
        if args.val != -1:
            permutation = dataset['permutation']
            split = dataset['split']
            val_adv_images = train_adv_images[permutation[:split]]
            train_adv_images = train_adv_images[permutation[split:]]
            val_adv_labels = [train_adv_labels[p] for p in permutation[:split]]
            train_adv_labels = [train_adv_labels[p] for p in permutation[split:]]
        adv_test_data = torch.load(test_path)
        test_adv_images = adv_test_data["adv"]
        test_adv_labels = adv_test_data["label"]        
    elif args.attack == "all" :
        for i in range(len(ATTACK_LIST)):
            _adv_dir = args.adv_dir + "{}/".format(ATTACK_LIST[i])
            train_path = _adv_dir + "train.pth" 
            test_path = _adv_dir + "test.pth"

            adv_train_data = torch.load(train_path)
            adv_test_data = torch.load(test_path)
            
            if i == 0 :
                train_adv_images = adv_train_data["adv"]
                train_adv_labels = adv_train_data["label"]
                test_adv_images = adv_test_data["adv"]
                test_adv_labels = adv_test_data["label"] 
                if args.val != -1:
                    permutation = dataset['permutation']
                    split = dataset['split']
                    val_adv_images = train_adv_images[permutation[:split]]
                    train_adv_images = train_adv_images[permutation[split:]]
                    val_adv_labels = [train_adv_labels[p] for p in permutation[:split]]
                    train_adv_labels = [train_adv_labels[p] for p in permutation[split:]]

            else :
                test_adv_images = np.concatenate((test_adv_images, adv_test_data["adv"]))
                test_adv_labels = np.concatenate((test_adv_labels, adv_test_data["label"]))
                
                if args.val != -1:
                    permutation = dataset['permutation']
                    split = dataset['split']
                    val_adv_images = np.concatenate((val_adv_images, adv_train_data["adv"][permutation[:split]]))
                    adv_train_data["adv"] = adv_train_data["adv"][permutation[split:]]
                    val_adv_labels = np.concatenate((val_adv_labels, [adv_train_data["label"][p] for p in permutation[:split]]))
                    adv_train_data["label"] = [adv_train_data["label"][p] for p in permutation[split:]]

                train_adv_images = np.concatenate((train_adv_images, adv_train_data["adv"]))
                train_adv_labels = np.concatenate((train_adv_labels, adv_train_data["label"]))


    else :
        raise ValueError("Unknown adversarial data")


        
    train_adv_set = list(zip(train_adv_images,
        train_adv_labels))

    
    if args.sample != 100 :
        n = len(train_adv_set) 
        n_sample = int(n * args.sample / 100)
        
        np.random.shuffle(train_adv_set)
        train_adv_set = train_adv_set[:n_sample]
        
    print("")
    print("Train Adv Attack Data: ", args.attack)
    print("Len: ", len(train_adv_set))
    print("")

    train_adv_batches = Batches(train_adv_set, args.batch_size, shuffle=shuffle, set_random_choices=False, num_workers=4)
    
    
    if args.val != -1:
        test_adv_set = list(zip(val_adv_images, val_adv_labels))
    else :
        test_adv_set = list(zip(test_adv_images,
            test_adv_labels))
        
    test_adv_batches = Batches(test_adv_set, args.batch_size, shuffle=False, num_workers=4)



    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

#     if args.model == 'PreActResNet18':
#         model = PreActResNet18()
#     elif args.model == 'WideResNet':
#         model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
#     else:
#         raise ValueError("Unknown model")
    model = resnet18(pretrained=True)


    model = nn.DataParallel(model).cuda()
    model.train()

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

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
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


    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(dirname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(dirname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_robust_acc = torch.load(os.path.join(dirname, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")
        
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

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, (batch, adv_batch) in enumerate(zip(train_batches, train_adv_batches)):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)
            
            adv_input = normalize(adv_batch['input'])
            y_adv = adv_batch['target']
            adv_input.requires_grad = True
            robust_output = model(adv_input)


            if args.mixup:
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = criterion(robust_output, y)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            output = model(normalize(X))
            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y_adv.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y_adv).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            

        train_time = time.time()


        
        # Evaluate on test data
        model.eval()
        test_loss = 0
        test_acc = 0
        test_n = 0
        
        test_robust_loss = 0
        test_robust_acc = 0
        test_robust_n = 0
        
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']
            
            clean_input = normalize(X)
            output = model(clean_input)
            loss = criterion(output, y)

            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            
        for i, batch in enumerate(test_adv_batches):                            
            adv_input = normalize(batch['input'])
            y = batch['target']

            robust_output = model(adv_input)
            robust_loss = criterion(robust_output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_robust_n += y.size(0)

        test_time = time.time()


        logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            epoch, train_time - start_time, test_time - train_time, lr,
            train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
            test_loss/test_n, test_acc/test_n, test_robust_loss/test_robust_n, test_robust_acc/test_robust_n)

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
            torch.save(model.state_dict(), os.path.join(dirname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(dirname, f'opt_{epoch}.pth'))

        # save best
        if test_robust_acc/test_robust_n > best_test_robust_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_robust_n,
                    'test_robust_loss':test_robust_loss/test_robust_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                }, os.path.join(dirname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc/test_n



if __name__ == "__main__":
    main()
