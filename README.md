# Adversarial Robustness 360 Toolbox (ART v0.10.0)
<p align="center">
  <img src="docs/images/art_logo.png?raw=true" width="200" title="ART logo">
</p>
<br />

[![Build Status](https://travis-ci.org/IBM/adversarial-robustness-toolbox.svg?branch=master)](https://travis-ci.org/IBM/adversarial-robustness-toolbox) [![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest) [![GitHub version](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox.svg)](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/context:python) [![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/alerts/)

This is a library dedicated to **adversarial machine learning**. Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. ART provides an implementation for many state-of-the-art methods for attacking and defending classifiers.

Documentation of ART: https://adversarial-robustness-toolbox.readthedocs.io

The library is still under development. Feedback, bug reports and extensions are highly appreciated. Get in touch with us on [Slack](https://ibm-art.slack.com) (invite [here](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTlkMWY3MzgyZDA4ZDdiNzUzY2NhMjc5YmFhZTYzZGYwNDM4YTE1ODhhNDYyNmFlMGFjNWY4ODgyM2EwYTFjYTc))!

## Supported Machine Learning Libraries
* Tensorflow (v1 and v2) (www.tensorflow.org)
* Keras (www.keras.io)
* PyTorch (www.pytorch.org)
* MXNet (https://mxnet.apache.org)
* Scikit-learn (www.scikit-learn.org)
* XGBoost (www.xgboost.ai)
* LightGBM (https://lightgbm.readthedocs.io)
* CatBoost (www.catboost.ai)
* GPy (https://sheffieldml.github.io/GPy/)

## Implemented Attacks, Defences, Detections, Metrics and Certifications

**Evasion Attacks:**
* HopSkipJump attack ([Chen et al., 2019](https://arxiv.org/abs/1904.02144))
* High Confidence Low Uncertainty adversarial examples ([Grosse et al., 2018](https://arxiv.org/abs/1812.02606)) 
* Projected gradient descent ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))
* NewtonFool ([Jang et al., 2017](http://doi.acm.org/10.1145/3134600.3134635))
* Elastic net attack ([Chen et al., 2017](https://arxiv.org/abs/1709.04114))
* Spatial transformations attack ([Engstrom et al., 2017](https://arxiv.org/abs/1712.02779))
* Query-efficient black-box attack ([Ilyas et al., 2017](https://arxiv.org/abs/1712.07113))
* Zeroth-order optimization attack ([Chen et al., 2017](https://arxiv.org/abs/1708.03999))
* Decision-based attack ([Brendel et al., 2018](https://arxiv.org/abs/1712.04248))
* Adversarial patch ([Brown et al., 2017](https://arxiv.org/abs/1712.09665))
* Decision tree attack ([Papernot et al., 2016](https://arxiv.org/pdf/1605.07277.pdf))
* C&amp;W `L_2` and `L_inf` attacks ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* Basic iterative method ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* Jacobian saliency map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Virtual adversarial method ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* Fast gradient method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))

**Defences:**
* Thermometer encoding ([Buckman et al., 2018](https://openreview.net/forum?id=S18Su--CW))
* Total variance minimization ([Guo et al., 2018](https://openreview.net/forum?id=SyJ7ClWCb))
* PixelDefend ([Song et al., 2017](https://arxiv.org/abs/1710.10766))
* Gaussian data augmentation ([Zantedeschi et al., 2017](https://arxiv.org/abs/1707.06728))
* Feature squeezing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Spatial smoothing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* JPEG compression ([Dziugaite et al., 2016](https://arxiv.org/abs/1608.00853))
* Label smoothing ([Warde-Farley and Goodfellow, 2016](https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf))
* Virtual adversarial training ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* Adversarial training ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))

**Detection of adversarial samples:**
* Basic detector based on inputs
* Detector trained on the activations of a specific layer
* Detector based on Fast Generalized Subset Scan ([Speakman et al., 2018](https://arxiv.org/pdf/1810.08676))

**Detectoion of poisoning attacks:**
* Detector based on activations analysis ([Chen et al., 2018](https://arxiv.org/abs/1811.03728))

**Robustness metrics, certifications and verification**:
* Clique Method Robustness Verification ([Hongge et al., 2019](https://arxiv.org/abs/1906.03849))
* Randomized Smoothing ([Cohen et al., 2019](https://arxiv.org/abs/1902.02918))
* CLEVER ([Weng et al., 2018](https://arxiv.org/abs/1801.10578))
* Loss sensitivity ([Arpit et al., 2017](https://arxiv.org/abs/1706.05394))
* Empirical robustness ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))

## Setup

### Installation with `pip`

The toolbox is designed and tested to run with Python 3. 
ART can be installed from the PyPi repository using `pip`:

```bash
pip install adversarial-robustness-toolbox
```

### Manual installation

The most recent version of ART can be downloaded or cloned from this repository:

```bash
git clone https://github.com/IBM/adversarial-robustness-toolbox
```

Install ART with following command from the project folder `art`:
```bash
pip install .
```

ART provides unit tests that can be run with the following command:

```bash
bash run_tests.sh
```

## Get Started with ART

Examples of using ART can be found in `examples` and `examples/README.md` provides an overview and additional 
information. It contains a minimal example for each machine learning framework. All examples cna be run with the 
following command:
```bash
python examples/<example_name>.py
```

More detailed examples are located in `notebooks` where Jupyter notebooks provide detailed tutorials. 

### Contributing

Adding new features, improving documentation, fixing bugs, or writing tutorials are all examples of helpful contributions. Furthermore, if you are publishing a new attack or defense, we strongly encourage you to add it to the Adversarial Robustness Toolbox so that others may evaluate it fairly in their own work.

Bug fixes can be initiated through GitHub pull requests. When making code contributions to the Adversarial Robustness Toolbox, we ask that you follow the `PEP 8` coding standard and that you provide unit tests for the new features.

This project uses [DCO](https://developercertificate.org/). Be sure to sign off your commits using the `-s` flag or adding `Signed-off-By: Name<Email>` in the commit message.

#### Example

```bash
git commit -s -m 'Add new feature'
```

## Citing ART

If you use ART for research, please consider citing the following reference paper:
```
@article{art2018,
    title = {Adversarial Robustness Toolbox v0.10.0},
    author = {Nicolae, Maria-Irina and Sinn, Mathieu and Tran, Minh~Ngoc and Buesser, Beat and Rawat, Ambrish and Wistuba, Martin and Zantedeschi, Valentina and Baracaldo, Nathalie and Chen, Bryant and Ludwig, Heiko and Molloy, Ian and Edwards, Ben},
    journal = {CoRR},
    volume = {1807.01069},
    year = {2018},
    url = {https://arxiv.org/pdf/1807.01069}
}
```
