.. adversarial-robustness-toolbox documentation master file, created by
   sphinx-quickstart on Fri Mar 23 17:02:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Adversarial Robustness Toolbox
=============================================

Adversarial Robustness Toolbox (ART) is a Python library supporting developers and researchers in defending Machine
Learning models (Deep Neural Networks, Gradient Boosted Decision Trees, Support Vector Machines, Random Forests,
Logistic Regression, Gaussian Processes, Decision Trees, Scikit-learn Pipelines, etc.) against adversarial threats and
helps making AI systems more secure and trustworthy. Machine Learning models are vulnerable to adversarial examples,
which are inputs (images, texts, tabular data, etc.) deliberately modified to produce a desired response by the Machine
Learning model. ART provides the tools to build and deploy defences and test them with adversarial attacks.

Defending Machine Learning models involves certifying and verifying model robustness and model hardening with approaches
such as pre-processing inputs, augmenting training data with adversarial samples, and leveraging runtime detection
methods to flag any inputs that might have been modified by an adversary. The attacks implemented in ART allow creating
adversarial attacks against Machine Learning models which is required to test defenses with state-of-the-art threat
models.

The code of ART is on `GitHub`_.

The library is under continuous development and feedback, bug reports and contributions are very welcome.

Supported Machine Learning Libraries
------------------------------------

* Tensorflow (v1 and v2) (https://www.tensorflow.org)
* Keras (https://www.keras.io)
* PyTorch (https://www.pytorch.org)
* MXNet (https://mxnet.apache.org)
* Scikit-learn (https://www.scikit-learn.org)
* XGBoost (https://www.xgboost.ai)
* LightGBM (https://lightgbm.readthedocs.io)
* CatBoost (https://www.catboost.ai)
* GPy (https://sheffieldml.github.io/GPy/)

Implemented Attacks, Defences, Detections, Metrics, Certifications and Verifications
------------------------------------------------------------------------------------

**Evasion Attacks:**

* HopSkipJump attack (`Chen et al., 2019`_)
* High Confidence Low Uncertainty adversarial examples (`Grosse et al., 2018`_)
* Projected gradient descent (`Madry et al., 2017`_)
* NewtonFool (`Jang et al., 2017`_)
* Elastic net attack (`Chen et al., 2017a`_)
* Spatial transformations attack (`Engstrom et al., 2017`_)
* Query-efficient black-box attack (`Ilyas et al., 2017`_)
* Zeroth-order optimization attack (`Chen et al., 2017b`_)
* Decision-based attack (`Brendel et al., 2018`_)
* Adversarial patch (`Brown et al., 2017`_)
* Decision tree attack (`Papernot et al., 2016b`_)
* Carlini & Wagner (C&W) L_2 and L_inf attacks (`Carlini and Wagner, 2016`_)
* Basic iterative method (`Kurakin et al., 2016`_)
* Jacobian saliency map (`Papernot et al., 2016a`_)
* Universal perturbation (`Moosavi-Dezfooli et al., 2016`_)
* DeepFool (`Moosavi-Dezfooli et al., 2015`_)
* Virtual adversarial method (`Miyato et al., 2015`_)
* Fast gradient method (`Goodfellow et al., 2014`_)

**Defences:**

* Thermometer encoding (`Buckman et al., 2018`_)
* Total variance minimization (`Guo et al., 2018`_)
* PixelDefend (`Song et al., 2017`_)
* Gaussian data augmentation (`Zantedeschi et al., 2017`_)
* Feature squeezing (`Xu et al., 2017`_)
* Spatial smoothing (`Xu et al., 2017`_)
* JPEG compression (`Dziugaite et al., 2016`_)
* Label smoothing (`Warde-Farley and Goodfellow, 2016`_)
* Virtual adversarial training (`Miyato et al., 2015`_)
* Adversarial training (`Szegedy et al., 2013`_)

**Robustness metrics, certifications and verifications:**

* Clique Method Robustness Verification (`Hongge et al., 2019`_)
* Randomized Smoothing (`Cohen et al., 2019`_)
* CLEVER (`Weng et al., 2018`_)
* Loss sensitivity (`Arpit et al., 2017`_)
* Empirical robustness (`Moosavi-Dezfooli et al., 2015`_)

**Detection of adversarial samples:**

* Basic detector based on inputs
* Detector trained on the activations of a specific layer
* Detector based on Fast Generalized Subset Scan (`Speakman et al., 2018`_)

**Detection of poisoning attacks:**

* Detector based on activations analysis (`Chen et al., 2018`_)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/setup
   guide/usage

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/attacks
   modules/classifiers
   modules/classifiers/classifiers_scikitlearn
   modules/data_generators
   modules/defences
   modules/detection
   modules/poison_detection
   modules/metrics
   modules/utils
   modules/utils_test
   modules/wrappers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https:github.com/IBM/adversarial-robustness-toolbox

.. _Chen et al., 2019: https://arxiv.org/abs/1904.02144
.. _Grosse et al., 2018: https://arxiv.org/abs/1812.02606

.. _Moosavi-Dezfooli et al., 2015: https://arxiv.org/abs/1511.04599
.. _Goodfellow et al., 2014: https://arxiv.org/abs/1412.6572
.. _Kurakin et al., 2016: https://arxiv.org/abs/1607.02533
.. _Madry et al., 2017: https://arxiv.org/abs/1706.06083
.. _Papernot et al., 2016a: https://arxiv.org/abs/1511.07528
.. _Moosavi-Dezfooli et al., 2016: https://arxiv.org/abs/1610.08401
.. _Carlini and Wagner, 2016: https://arxiv.org/abs/1608.04644
.. _Jang et al., 2017: http://doi.acm.org/10.1145/3134600.3134635
.. _Chen et al., 2017a: https://arxiv.org/abs/1709.04114
.. _Chen et al., 2017b: https://arxiv.org/abs/1708.03999
.. _Engstrom et al., 2017: https://arxiv.org/abs/1712.02779
.. _Ilyas et al., 2017: https://arxiv.org/abs/1712.07113
.. _Xu et al., 2017: http://arxiv.org/abs/1704.01155
.. _Warde-Farley and Goodfellow, 2016: https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf
.. _Szegedy et al., 2013: http://arxiv.org/abs/1312.6199
.. _Miyato et al., 2015: https://arxiv.org/abs/1507.00677
.. _Zantedeschi et al., 2017: https://arxiv.org/abs/1707.06728
.. _Buckman et al., 2018: https://openreview.net/forum?id=S18Su--CW
.. _Guo et al., 2018: https://openreview.net/forum?id=SyJ7ClWCb
.. _Dziugaite et al., 2016: https://arxiv.org/abs/1608.00853
.. _Song et al., 2017: https://arxiv.org/abs/1710.10766
.. _Chen et al., 2018: https://arxiv.org/abs/1811.03728
.. _Weng et al., 2018: https://arxiv.org/abs/1801.10578
.. _Arpit et al., 2017: https://arxiv.org/abs/1706.05394
.. _Brendel et al., 2018: https://arxiv.org/abs/1712.04248
.. _Brown et al., 2017: https://arxiv.org/abs/1712.09665

.. _Speakman et al., 2018: https://arxiv.org/pdf/1810.08676
.. _Papernot et al., 2016b: https://arxiv.org/abs/1605.07277

.. _Cohen et al., 2019: https://arxiv.org/abs/1902.02918
.. _Hongge et al., 2019: https://arxiv.org/abs/1906.03849
