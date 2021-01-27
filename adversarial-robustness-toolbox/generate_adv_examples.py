
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

sys.path.insert(0,'/workspace/attack-analysis')
from models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
## data preprocessing
#####################

cifar_mu = np.ones((3, 32, 32)).astype(np.float32)
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465
cifar_mu.astype(np.float32)

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

cifar_std = np.ones((3, 32, 32)).astype(np.float32)
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616
cifar_std.astype(np.float32)


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

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
    parser.add_argument('--output-dir', default='../adv_examples/', type=str)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='autoattack', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=200, type=int)

    return parser.parse_args()



if __name__ == "__main__" :

    args = get_args()
    
    dirname = args.output_dir + args.attack + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    train_path =  "{}train.pth".format(dirname)
    test_path = "{}test.pth".format(dirname)
    
#     if os.path.exists(train_path):
#         print("Adversarial examples already exist at {}".format(train_path))
#         print("Please remove it to generate the new one!")
#         sys.exit()
    

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Step 1: Load the CIFAR10 dataset
    dataset = cifar10(args.data_dir)
    x_train = (dataset['train']['data']/255.)
    x_test = (dataset['test']['data']/255.)
    x_train = transpose(x_train).astype(np.float32)
    x_test = transpose(x_test).astype(np.float32)
    
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']
    
    # Step 2: Load the pretrained model

    model = resnet18(pretrained=True)
    model.cuda()
    model.eval()
    
    # Step 2a: Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    lr_max = 1e-2
    weight_decay = 1e-2
    params = model.parameters()
    optimizer = torch.optim.SGD(params,lr=lr_max, momentum=0.9, weight_decay=weight_decay)
    
    min_pixel_value=0
    max_pixel_value=1


    # Step 3: Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        preprocessing=(cifar_mu, cifar_std),
    )
    
    # Step 5: Evaluate the ART classifier on benign test examples
    
#     normalized_x_test = normalize(x_test)

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
#     accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("=== Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    epsilon = (8. / 255.)
    batch_size = 1000
    
#     print("Epsilon: ", epsilon)
    
    
    attack = None
    if args.attack == "autoattack" :
        attack = AutoAttack(estimator=classifier, eps=epsilon, eps_step=0.75, batch_size=batch_size)
        # the parameter is obtained from https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py
    elif args.attack == "autopgd" :
        attack = AutoProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.75, batch_size=batch_size)
        # the parameter is obtained from https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py
    elif args.attack == "boundaryattack" :
        attack = BoundaryAttack(estimator=classifier, targeted=False)
        # the parameter is obtained from
    elif args.attack == "brendelbethge" :
        attack = BrendelBethgeAttack(estimator=classifier, batch_size=batch_size)
        # the parameter is obtained from
    elif args.attack == "cw" :
        attack = CarliniLInfMethod(classifier=classifier, eps=epsilon, batch_size=batch_size)
        # the parameter is obtained from
    elif args.attack == "deepfool" :
        attack = DeepFool(classifier=classifier, max_iter=50, epsilon=0.02, batch_size=batch_size)
        # the parameter is obtained from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
    elif args.attack == "elasticnet" :
        attack = ElasticNet(classifier=classifier, beta=1e-2, batch_size=batch_size)
        # the parameter is obtained from https://github.com/ysharma1126/EAD_Attack/blob/master/test_attack.py#L367
    elif args.attack == "fgsm" :
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        # the parameter is obtained from
    elif args.attack == "hopskipjump" :
        attack = HopSkipJump(classifier=classifier)
        # the parameter is obtained from
    elif args.attack == "bim" :
        attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=1./255., batch_size=batch_size)
        # the parameter is obtained from https://arxiv.org/pdf/1607.02533.pdf
        # "we changed the value of each pixel only by 1 on each step."
    elif args.attack == "pgd" :
        attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=epsilon, eps_step=2./255., max_iter=40, batch_size=batch_size)
        # the parameter is obtained from 
    elif args.attack == "newtonfool" :
        attack = NewtonFool(classifier=classifier, batch_size=batch_size)
        # the parameter is obtained from
    elif args.attack == "pixelattack" :
        attack = PixelAttack(classifier=classifier)
        # the parameter is obtained from
    elif args.attack == "thresholdattack" :
        attack = ThresholdAttack(classifier=classifier)
        # the parameter is obtained from
    elif args.attack == "jsma" :
        attack = SaliencyMapMethod(classifier=classifier, batch_size=batch_size)
        # the parameter is obtained from
    elif args.attack == "shadowattack" :
        attack = ShadowAttack(estimator=classifier, batch_size=1)
        # the parameter is obtained from
    elif args.attack == "spatialtransformation" :
        attack = SpatialTransformation(classifier=classifier, max_translation=3, max_rotation=30)
        # the parameter is obtained from
    elif args.attack == "squareattack" :
        attack = SquareAttack(estimator=classifier, eps=epsilon, batch_size=batch_size)
        # the parameter is obtained from https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py
#     elif args.attack == "universalperturbation" :
#         attack = UniversalPerturbation(classifier=classifier, batch_size=batch_size)
        # the parameter is obtained from 
    elif args.attack == "wasserstein" :
        attack = Wasserstein(estimator=classifier, batch_size=1024)
        # the parameter is obtained from
    elif args.attack == "zoo" :
        attack = ZooAttack(classifier=classifier, nb_parallel=1024, batch_size=256)
        # the parameter is obtained from
    else :
        raise ValueError("Unknown model")

        
    x_train_adv = attack.generate(x=x_train, y=y_train)
    torch.save({"adv": x_train_adv, "label":y_train }, train_path)
    print("Adversarial examples from train data is saved at {}".format(train_path))

    x_test_adv = attack.generate(x=x_test, y=y_test)
    torch.save({"adv": x_test_adv, "label":y_test }, test_path)
    print("Adversarial examples from test data is saved at {}".format(test_path))
    

    # Step 7: Evaluate the ART classifier on adversarial test examples
    
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
# #     accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))