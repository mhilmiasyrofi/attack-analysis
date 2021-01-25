# Attack Analysis

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti
- Cuda: 10.1
- Python: 3.7
- PyTorch: 1.5.1
- Torchvision: 0.6.0
- Scikit-Learn
- Pandas
- Seaborn

Docker: pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

## Model Preparation
Please download the ResNet18 pretrained model (on CIFAR10) from this [Google Drive](). The downloaded zip contain another things needed in our experiment, e.g. adversarial examples, and also contain the result of our experiment, e.g. trained models.
Please extract the downloaded folder inside the `<root-project>`. At this step, you need `<root-project>/models/`.

## Adversarial Examples Generation 

In this project, we use 11 adversarial attacks presented in the table below

"autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack"

|Name|Type|
|AutoAttack|White-box|
|AutoPGD|White-box|
|BIM|White-box|
|CW|White-box|
|DeepFool|White-box|
|FGSM|White-box|
|NewtonFool|White-box|
|PGD|White-box|
|PixelAttack|Black-box|
|SpatialTransformation|Black-box|
|SquareAttack|Black-box|

We use [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to generate adversarial examples.
The detail is [here](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/adversarial-robustness-toolbox).


## Adversarial Training

In our experiment, we use 3 adversarial training techniques, [BagOfTricks](https://arxiv.org/abs/2010.00467), [AT](https://arxiv.org/pdf/2002.11569.pdf), [AWP](https://arxiv.org/pdf/2004.05884.pdf).

### BagOfTricks Related Experiment

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/Bag-of-Tricks-for-AT)

### AT Related Experiment

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/robust_overfitting)

### AWP Related Experiment

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/AWP)

