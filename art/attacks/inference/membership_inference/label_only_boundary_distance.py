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
This module implements the classifier `BlackBoxClassifier` for black-box classifiers.
"""
import logging
from typing import Optional

import numpy as np

from art.attacks.attack import InferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

logger = logging.getLogger(__name__)


class LabelOnlyDecisionBoundary(InferenceAttack):
    """
    Implementation of Label-Only Inference Attack based on Decision Boundary.
    """

    attack_params = InferenceAttack.attack_params + ["distance_cutoff_tau",]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE", distance_cutoff_tau: Optional[float] = None):
        """
        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.

        :param estimator: A trained classification estimator.
        :param distance_cutoff_tau: Cut-off distance for decision boundary. Samples with boundary distances larger than
                                    cut-off are considered members of the training dataset.
        """
        super().__init__(estimator=estimator)
        self.distance_cutoff_tau = distance_cutoff_tau
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        from art.attacks.evasion.hop_skip_jump import HopSkipJump

        hsj = HopSkipJump(classifier=self.estimator, **kwargs)
        x_adv = hsj.generate(x=x, y=y)

        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)

        y_pred = self.estimator.predict(x=x)

        distance[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)] = 0

        is_member = np.where(distance > self.distance_cutoff_tau, 1, 0)

        return is_member

    def calibrate_distance_cutoff(self, classifier_train, x_train, y_train, x_test, y_test, **kwargs):
        from art.attacks.evasion.hop_skip_jump import HopSkipJump

        hsj = HopSkipJump(classifier=classifier_train, **kwargs)

        x_train_adv = hsj.generate(x=x_train, y=y_train)
        x_test_adv = hsj.generate(x=x_test, y=y_test)

        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)

        y_train_pred = self.estimator.predict(x=x_train)
        y_test_pred = self.estimator.predict(x=x_test)

        distance_train[np.argmax(y_train_pred, axis=1) != np.argmax(y_train, axis=1)] = 0
        distance_test[np.argmax(y_test_pred, axis=1) != np.argmax(y_test, axis=1)] = 0

        tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / 100

        acc_max = 0.0
        distance_cutoff_tau = 0.0

        for i_tau in range(1, 100):

            is_member_train = np.where(distance_train > i_tau * tau_increment, 1, 0)
            is_member_test = np.where(distance_test > i_tau * tau_increment, 1, 0)

            acc = (np.sum(is_member_train) + (is_member_test.shape[0] - np.sum(is_member_test))) / (
                        is_member_train.shape[0] + is_member_test.shape[0])

            print(i_tau, i_tau * tau_increment, acc)

            if acc > acc_max:
                distance_cutoff_tau = i_tau * tau_increment
                acc_max = acc

        self.distance_cutoff_tau = distance_cutoff_tau
