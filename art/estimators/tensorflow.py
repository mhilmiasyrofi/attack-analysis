# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the abstract estimators `TensorFlowEstimator` and `TensorFlowV2Estimator` for TensorFlow models.
"""
import logging

from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin

logger = logging.getLogger(__name__)


class TensorFlowEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow models.
    """


class TensorFlowV2Estimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow v2 models.
    """
