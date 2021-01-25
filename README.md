# Bag of Tricks for Adversarial Training
A part of code for adversarial training originated from [Bag of Tricks for Adversarial Training](https://arxiv.org/abs/2010.00467) 

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti
- Cuda: 10.1
- Python: 3.7
- PyTorch: 1.5.1
- Torchvision: 0.6.0
Docker: pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

*Go to `cifar10` folder*


## RQ1

Running command for adversarial training
```
bash train-rq1.sh
```
This script will generate trained models that is saved at `<fpath>`

Run evaluation using the trained models
```
bash eval-rq1.sh
```
This script will .... generate log on `<fpath>`

Clustering analysis: `clustering-analysis-rq1.ipynb`


## RQ2

Best performing model analysis: `best-performing-model-analysis.ipynb`


## RQ4

Running command for adversarial training
```
bash train-rq1.sh
```
This script will generate trained models that is saved at `<fpath>`

Run evaluation using the trained models
```
bash eval-rq1.sh
```
This script will .... generate log on `<fpath>`

Clustering analysis: `clustering-analysis-rq1.ipynb`
