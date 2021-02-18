# Attack Analysis

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti
- Cuda: 10.1
- Python: 3.7
- PyTorch: 1.5.1
- Torchvision: 0.6.0

Docker: pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

### Some Useful Tips for Development

#### Maintaining the Session in Server
[Why should you use tmux?](https://medium.com/@brindelle/why-should-you-learn-tmux-7a55cfb5668f). *TL;DR*: Instead of keeping track of many windows yourself, you can use tmux to create, organize, and navigate between them. Even more importantly, tmux lets you detach from and re-attach sessions, so that you can leave your terminal sessions running in the background and resume them later.


#### Docker Preparationo 

```
docker run -it --rm --name gpu0-at -v <path to the folder>/attack-analysis/:/workspace/attack-analysis/ --gpus '"device=0"' pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

pip install -r requirements.txt
```

#### Writing Code on Your Local PC with Jupyter Lab running in The Remote Server
* add port forwarding when running the docker with `-p 8888:8888` -> you can try another port if it's already used by another process
* run jupyter notebook `jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root`
* open your jupyter lab from your favorite browser on your local PC `<your serverip>:8888`



## Model Preparation
Please download the ResNet18 pretrained model (on CIFAR10) from this [Google Drive](). The downloaded zip contains another things needed in our experiment, e.g. adversarial examples, and also contain some of the results from our experiment, e.g. trained models.
Please extract the downloaded folder inside the `<root-project>`. At this step, you need `<root-project>/models/`.

*TODO:* add google drive link

## Adversarial Examples Generation 

In this project, we use 11 adversarial attacks presented in the table below

|Name|Type|
|---|---|
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

In our experiment, we use 3 adversarial training techniques, [BagOfTricks](https://arxiv.org/abs/2010.00467), [AWP](https://arxiv.org/pdf/2004.05884.pdf), [AT](https://arxiv.org/pdf/2002.11569.pdf).

### BagOfTricks Related Experiment - RQ1, RQ3, RQ4

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/Bag-of-Tricks-for-AT)

### AWP Related Experiment - RQ2

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/AWP)

### AT Related Experiment - RQ2

Please directly check [the folder](https://github.com/mhilmiasyrofi/attack-analysis/tree/master/AT)


