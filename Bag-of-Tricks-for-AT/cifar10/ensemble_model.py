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
    parser.add_argument('--centroids', default='pixelattack_spatialtransformation_autopgd')
    parser.add_argument('--noise-predictor', default='maxlr0.05_wd0.0001_ls0.3')
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

    parser.add_argument('--batch-size', default=1, type=int) #batch size

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
    names = args.model + '_'  + args.centroids + '_' + args.noise_predictor + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)

    print('File name: ', names)
    return names

def main():
    args = get_args()
    base_dir = 'ensemble_models/'
    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = base_dir + names
    else:
        args.fname = base_dir + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)
        
    eval_dir = args.fname + '/eval/all/'

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

    ATTACK_LIST = ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "newtonfool", "pgd", "pixelattack", "spatialtransformation", "squareattack"]
#     ATTACK_LIST = ["autopgd"]
    
    print("Load test data...")

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
            train_adv_images = np.concatenate((train_adv_images, adv_train_data["adv"]))
            train_adv_labels = np.concatenate((train_adv_labels, adv_train_data["label"]))
            test_adv_images = np.concatenate((test_adv_images, adv_test_data["adv"]))
            test_adv_labels = np.concatenate((test_adv_labels, adv_test_data["label"]))

        test_adv_images_on_train = train_adv_images
        test_adv_labels_on_train = train_adv_labels
        test_adv_images_on_test = test_adv_images
        test_adv_labels_on_test = test_adv_labels
        
    print("Add to dataloader...")

    
    test_adv_on_train_set = list(zip(test_adv_images_on_train,
        test_adv_labels_on_train))
    
    test_adv_on_train_batches = Batches(test_adv_on_train_set, args.batch_size, shuffle=False, set_random_choices=False, num_workers=0)
    
    test_adv_on_test_set = list(zip(test_adv_images_on_test,
        test_adv_labels_on_test))
        
    test_adv_on_test_batches = Batches(test_adv_on_test_set, args.batch_size, shuffle=False, num_workers=0)
    
    print("Load model...")

    centroids = args.centroids.split("_")
    
    models = {}
    for c in centroids :
        models[c] = resnet18()
        models[c].cuda()
        model_path = "trained_models/backup/resnet18_{}_piecewise_eps8_bs128_maxlr0.1_ls0.3_BNeval/model_best.pth".format(c)
        models[c].load_state_dict(torch.load(model_path)["state_dict"])
        models[c].eval()
        
    noise_predictor = resnet18(num_classes=len(centroids))
    noise_predictor.cuda()
    n = len(centroids)
    noise_predictor_path = os.path.join("noise_predictor/resnet18_{}_{}_piecewise_eps8_bs128_{}_BNeval/".format(n, args.centroids, args.noise_predictor), f'model_best.pth')
    noise_predictor.load_state_dict(torch.load(noise_predictor_path)["state_dict"])
    noise_predictor.eval()

    
    test_adv_test_acc = 0
    test_adv_test_n = 0
    y_adv = np.array([])
    y_adv_pred = np.array([])

#     test_adv_train_loss = 0
#     test_adv_train_acc = 0
#     test_adv_train_n = 0

    id2centroid = {}
    i = 0
    for c in centroids :
        id2centroid[i] = c
        i += 1

    print("Predicting...")
    

    for i, batch in enumerate(test_adv_on_test_batches):
        adv_input = normalize(batch['input'])
        
        cluster_output = noise_predictor(adv_input)
#         print("Cluster output: ", cluster_output)
        cluster_id = int(cluster_output.max(1)[1][0])
#         print("Cluster id: ", cluster_id)
        
        y = batch['target']

#         model = models["autopgd"]        
        model = models[id2centroid[cluster_id]]


        cross_robust_output = model(adv_input)

        test_adv_test_acc += (cross_robust_output.max(1)[1] == y).sum().item()
        test_adv_test_n += y.size(0)

        y_adv = np.append(y_adv, y.cpu().numpy())
        y_adv_pred = np.append(y_adv_pred, cross_robust_output.max(1)[1].cpu().numpy())

        
#     for i, batch in enumerate(test_adv_on_train_batches):
#         adv_input = normalize(batch['input'])
#         y = batch['target']

#         cross_robust_output = model(adv_input)
#         cross_robust_loss = criterion(cross_robust_output, y)

#         test_adv_train_loss += cross_robust_loss.item() * y.size(0)
#         test_adv_train_acc += (cross_robust_output.max(1)[1] == y).sum().item()
#         test_adv_train_n += y.size(0)

    test_time = time.time()

    logger.info("Test Robust Acc on Test")
    logger.info('%.4f', 
                test_adv_test_acc/test_adv_test_n)

#     logger.info("Test Acc \tTest Robust Acc on Test \tTest Robust Acc on Train")
#     logger.info('%.4f \t\t %.4f \t\t %.4f', 
#                 test_acc/test_n,
#                 test_adv_test_acc/test_adv_test_n,
#                 test_adv_train_acc/test_adv_train_n)

    
#     y_original = y_original.astype(np.int)
#     y_original_pred = y_original_pred.astype(np.int)

#     logger.info("Y_original")
#     logger.info(y_original)
#     np.savetxt(os.path.join(eval_dir, "Y_original.txt"), y_original,  fmt='%i')
    
#     logger.info("Y_original_pred")
#     logger.info(y_original_pred)
#     np.savetxt(os.path.join(eval_dir, "Y_original_pred.txt"), y_original_pred, fmt='%i')


    y_adv = y_adv.astype(np.int)
    y_adv_pred = y_adv_pred.astype(np.int)    
    
    logger.info("Y_adv")
    logger.info(y_adv)
    np.savetxt(os.path.join(eval_dir, "Y_adv.txt"), y_adv, fmt='%i')
    
    logger.info("Y_adv_pred")
    logger.info(y_adv_pred)
    np.savetxt(os.path.join(eval_dir, "Y_adv_pred.txt"), y_adv_pred, fmt='%i')
    
    
if __name__ == "__main__":
    main()
