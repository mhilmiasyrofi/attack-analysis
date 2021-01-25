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


import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils import *
from utils_awp import AdvWeightPerturb

from models import *
from constant import TOOLBOX_ADV_ATTACK_LIST


mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


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
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--centroids', default='pixelattack_spatialtransformation_autopgd')
    parser.add_argument('--noise-predictor', default='maxlr0.05_wd0.0001_ls0.3')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='ensemble_models', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty
    
    fname = args.fname + "/" 
    if not os.path.exists(fname):
        os.makedirs(fname)
        
    fname += args.centroids + "/"
    if not os.path.exists(fname):
        os.makedirs(fname)

    
    eval_dir = fname + "eval/all/"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(eval_dir, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    train_adv_images = None
    train_adv_labels = None
    test_adv_images = None
    test_adv_labels = None

    ATTACK_LIST = ["autoattack", "autopgd", "bim", "cw", "deepfool", "fgsm", "newtonfool", "pgd", "pixelattack", "spatialtransformation", "squareattack"]
    
#     ATTACK_LIST = ["autoattack"]
    
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
            
    train_adv_set = list(zip(train_adv_images,
        train_adv_labels))
    
    train_adv_batches = Batches(train_adv_set, args.batch_size, shuffle=True, set_random_choices=False, num_workers=4)
    
    test_adv_set = list(zip(test_adv_images,
        test_adv_labels))
        
    test_adv_batches = Batches(test_adv_set, args.batch_size, shuffle=False, num_workers=4)
    
#     model = resnet18(pretrained=True)
#     model = nn.DataParallel(model).cuda()
#     model.eval()
    
    
    print("Load model...")

    centroids = args.centroids.split("_")
    
    models = {}
    for c in centroids :
        models[c] = resnet18()
        models[c].cuda()
        model_path = "trained_models/{}/model_best.pth".format(c)
        models[c] = nn.DataParallel(models[c]).cuda()
        models[c].load_state_dict(torch.load(model_path)["state_dict"])
        models[c].eval()
        
    noise_predictor = resnet18(num_classes=len(centroids))
    noise_predictor.cuda()
    n = len(centroids)
    noise_predictor_path = os.path.join("noise_predictors/resnet18_{}_{}_piecewise_eps8_bs128_{}_BNeval/".format(n, args.centroids, args.noise_predictor), f'model_best.pth')
    noise_predictor.load_state_dict(torch.load(noise_predictor_path)["state_dict"])
    noise_predictor.eval()
    
    id2centroid = {}
    i = 0
    for c in centroids :
        id2centroid[i] = c
        i += 1
    
    test_robust_acc = 0
    test_robust_n = 0
    
#     Y_original = np.array([])
#     Y_original_pred = np.array([])

    Y_adv = np.array([])
    Y_adv_pred = np.array([])

    
    for i, adv_batch in enumerate(test_adv_batches):
        X_adv, y_adv = normalize(adv_batch["input"]), adv_batch["target"]
        
        cluster_output = noise_predictor(X_adv)
#         print("Cluster output: ", cluster_output)
        cluster_id = int(cluster_output.max(1)[1][0])
#         print("Cluster id: ", cluster_id)

#         model = models["autopgd"]        
        model = models[id2centroid[cluster_id]]

        robust_output = model(X_adv)
    
        test_robust_acc += (robust_output.max(1)[1] == y_adv).sum().item()
        test_robust_n += y_adv.size(0)

        Y_adv = np.append(Y_adv, y_adv.cpu().numpy())
        Y_adv_pred = np.append(Y_adv_pred, robust_output.max(1)[1].cpu().numpy())
    
    
    logger.info("Test Robust Acc")
    logger.info('%.4f', 
                test_robust_acc/test_robust_n)
    
#     Y_original = Y_original.astype(np.int)
#     Y_original_pred = Y_original_pred.astype(np.int)

#     print("Y_original")
#     print(Y_original)
#     np.savetxt(os.path.join(eval_dir, "Y_original.txt"), Y_original,  fmt='%i')

#     print("Y_original_pred")
#     print(Y_original_pred)
#     np.savetxt(os.path.join(eval_dir, "Y_original_pred.txt"), Y_original_pred, fmt='%i')


    Y_adv = Y_adv.astype(np.int)
    Y_adv_pred = Y_adv_pred.astype(np.int)    

    print("Y_adv")
    print(Y_adv)
    np.savetxt(os.path.join(eval_dir, "Y_adv.txt"), Y_adv, fmt='%i')

    print("Y_adv_pred")
    print(Y_adv_pred)
    np.savetxt(os.path.join(eval_dir, "Y_adv_pred.txt"), Y_adv_pred, fmt='%i')


if __name__ == "__main__":
    main()