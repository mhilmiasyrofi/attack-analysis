# Adversarial Robustness Toolbox (ART) v1.3
<p align="center">
  <img src="docs/images/art_lfai.png?raw=true" width="467" title="ART logo">
</p>
<br />

[![Build Status](https://travis-ci.org/IBM/adversarial-robustness-toolbox.svg?branch=master)](https://travis-ci.org/IBM/adversarial-robustness-toolbox)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox.svg)](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/alerts/)
[![codecov](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adversarial-robustness-toolbox)](https://pypi.org/project/adversarial-robustness-toolbox/)
[![slack-img](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://ibm-art.slack.com/)

[中文README请按此处](README-cn.md)

Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides tools that enable
developers and researchers to evaluate, defend, certify and verify Machine Learning models and applications against the
adversarial threats of Evasion, Poisoning, Extraction, and Inference. ART supports all popular machine learning frameworks
(TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types
(images, tables, audio, video, etc.) and machine learning tasks (classification, object detection, generation,
certification, etc.).

<p align="center">
  <img src="docs/images/adversarial_threats_attacker.png?raw=true" width="400" title="ART logo">
  <img src="docs/images/adversarial_threats_art.png?raw=true" width="400" title="ART logo">
</p>
<br />

## Learn more

| **[Get Started][get-started]**     | **[Documentation][documentation]**     | **[Contributing][contributing]**           |
|-------------------------------------|-------------------------------|-----------------------------------|
| - [Installation][installation]<br>- [Examples](examples/README.md)<br>- [Notebooks](notebooks/README.md) | - [Attacks][attacks]<br>- [Defences][defences]<br>- [Estimators][estimators]<br>- [Metrics][metrics]<br>- [Technical Documentation](https://adversarial-robustness-toolbox.readthedocs.io) | - [Slack](https://ibm-art.slack.com), [Invitation](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg)<br>- [Contributing](CONTRIBUTING.md)<br>- [Roadmap][roadmap]<br>- [Citing][citing] |

[get-started]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Get-Started
[attacks]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/ART-Attacks
[defences]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/ART-Defences
[estimators]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/ART-Estimators
[metrics]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/ART-Metrics
[contributing]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Contributing
[documentation]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Documentation
[installation]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Get-Started#setup
[roadmap]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Roadmap
[citing]: https://github.com/IBM/adversarial-robustness-toolbox/wiki/Contributing#citing-art

The library is under continuous development. Feedback, bug reports and contributions are very welcome!
