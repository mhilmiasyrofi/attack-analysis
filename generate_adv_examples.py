
"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import os
import argparse
import sys
import time
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from art.config import ART_NUMPY_DTYPE
from art.attacks.evasion import *
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset


from models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def normalize(X):
    return (X - mu)/std

def np_normalize(x, mean=cifar10_mean, std=cifar10_std):
    return (x - mean)/std

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='autoattack', type=str, choices=["autoattack", "apgd", "boundaryattack", "brendelbethge", "deepfool", "bim", "elasticnet", "pgd", "jsma", "shadowattack", "squareattack", "wasserstein", "fgm"])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=200, type=int)

    return parser.parse_args()



if __name__ == "__main__" :

    args = get_args()
    
    dirname = "adv_examples/" + args.attack + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    train_path =  "{}train.pth".format(dirname)
    test_path = "{}test.pth".format(dirname)
    
    if os.path.exists(train_path):
        print("Adversarial examples already exist at {}".format(train_path))
        print("Please remove it to generate the new one!")
        sys.exit()
    

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Step 1: Load the CIFAR10 dataset
    dataset = cifar10(args.data_dir)
    x_train = (dataset['train']['data']/255.)
    x_test = (dataset['test']['data']/255.)
    x_train = np_normalize(x_train)
    x_test = np_normalize(x_test)
    x_train = transpose(x_train).astype(np.float32)
    x_test = transpose(x_test).astype(np.float32)
    
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']
    
#     print("train shape: ", x_train.shape)
#     print("test shape: ", x_test.shape)
#     print("y train shape: ", y_train.shape)
#     print("y test shape: ", y_test.shape)
#     print(y_test)

    # Step 2: Load the pretrained model

    model = resnet18(pretrained=True)
    model.cuda()
    model.eval()
    
    # Step 2a: Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
#     lr_max = 0.1
#     weight_decay = 5e-4
    lr_max = 1e-2
    weight_decay = 1e-2
    params = model.parameters()
    optimizer = torch.optim.SGD(params,lr=lr_max, momentum=0.9, weight_decay=weight_decay)
    
    min_pixel_value=0
    max_pixel_value=1
#     print("min_pixel_value: ", min_pixel_value)
#     print("max_pixel_value: ", max_pixel_value)


    # Step 3: Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        preprocessing=None,
    )
    
    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
#     accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("=== Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    epsilon = (8. / 255.)
    eps_step = (8. / 255. / 3.)

    attack = None
    if args.attack == "autoattack" :
        attack = AutoAttack(estimator=classifier, eps=epsilon, eps_step=eps_step)
    elif args.attack == "apgd" :
        attack = AutoProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=eps_step)
    elif args.attack == "boundaryattack" :
        attack = BoundaryAttack(estimator=classifier, eps=epsilon)
    elif args.attack == "brendelbethge" :
        attack = BrendelBethgeAttack(estimator=classifier, eps=epsilon)
    elif args.attack == "deepfool" :
        attack = DeepFool(estimator=classifier, eps=epsilon)
    elif args.attack == "bim" :
        attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=eps_step)
    elif args.attack == "elasticnet" :
        attack = ElasticNet(classifier=classifier)
    elif args.attack == "pgd" :
        attack = ProjectedGradientDescent(estimator=classifier, eps_step=eps_step)
    elif args.attack == "jsma" :
        attack = SaliencyMapMethod(classifier=classifier)
    elif args.attack == "shadowattack" :
        attack = ShadowAttack(estimator=classifier, eps=epsilon)
    elif args.attack == "squareattack" :
        attack = SquareAttack(estimator=classifier, eps=epsilon)
    elif args.attack == "wasserstein" :
        attack = Wasserstein(estimator=classifier, eps=epsilon, eps_step=eps_step)
    elif args.attack == "fgm" :
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    else :
        raise ValueError("Unknown model")
#     elif args.attack = "" :
#         attack = (estimator=classifier, eps=epsilon)
#     elif args.attack = "" :
#         attack = (estimator=classifier, eps=epsilon)
#     elif args.attack = "" :
#         attack = (estimator=classifier, eps=epsilon)
        
    x_train_adv = attack.generate(x=x_train)
    torch.save({"adv": x_train_adv, "label":y_train }, train_path)
    print("Adversarial examples from train data is saved at {}".format(train_path))

    x_test_adv = attack.generate(x=x_test)
    torch.save({"adv": x_test_adv, "label":y_test }, test_path)
    print("Adversarial examples from test data is saved at {}".format(test_path))
    
    
#     data = torch.load("_test.pth")
#     x_test_adv = data["adv"]

#     # Step 7: Evaluate the ART classifier on adversarial test examples

#     predictions = classifier.predict(x_test_adv)
#     accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
# #     accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#     print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))