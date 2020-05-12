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
This module implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

| Paper link: https://arxiv.org/abs/1902.07906
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class Wasserstein(EvasionAttack):
    """
    Implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

    | Paper link: https://arxiv.org/abs/1902.07906
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "regularization",
        "p",
        "kernel_size",
        "alpha",
        "norm",
        "ball",
        "epsilon",
        "epsilon_factor",
        "max_iter",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        estimator,
        targeted=False,
        regularization=3000,
        p=2,
        kernel_size=5,
        alpha=0.1,
        norm='wasserstein',
        ball='wasserstein',
        epsilon=0.001,
        epsilon_factor=1.17,
        max_iter=400,
        batch_size=1,
    ):
        """
        Create a Wasserstein attack instance.

        :param estimator: A trained estimator.
        :type estimator: :class:`.BaseEstimator`
        """
        super(Wasserstein, self).__init__(estimator=estimator)

        kwargs = {
            "targeted": targeted,
            "regularization": regularization,
            "p": p,
            "kernel_size": kernel_size,
            "alpha": alpha,
            "norm": norm,
            "ball": ball,
            "epsilon": epsilon,
            "epsilon_factor": epsilon_factor,
            "max_iter": max_iter,
            "batch_size": batch_size,
        }
        Wasserstein.set_params(self, **kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        # Compute the cost matrix if needed
        cost_matrix = kwargs.get("cost_matrix")
        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix()

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x[batch_index_1: batch_index_2]
            batch_labels = targets[batch_index_1: batch_index_2]

            x_adv[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, cost_matrix)

        return x_adv

    def _compute_cost_matrix(self):
        """
        Compute the default cost matrix.

        :return: The cost matrix.
        :rtype: `np.ndarray`
        """
        center = self.kernel_size // 2
        cost_matrix = np.zeros((self.kernel_size, self.kernel_size))

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                cost_matrix[i, j] = (abs(i - center) ** 2 + abs(j - center) ** 2) ** (self.p / 2)

        return cost_matrix

    def _generate_batch(self, x, targets, cost_matrix):
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :type targets: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        adv_x = x.copy()
        for _ in range(self.max_iter):
            adv_x = self._compute(
                adv_x,
                x,
                targets,
                self.eps,
                self.eps_step,
                cost_matrix
            )

        return adv_x

    def _compute(self, x, x_init, y, eps, eps_step, cost_matrix):
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :type x: `np.ndarray`
        :param x_init: An array with the original inputs.
        :type x_init: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        # Compute and apply perturbation
        x_adv = self._compute_apply_perturbation(x, y, eps_step, cost_matrix)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm, cost_matrix)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _compute_apply_perturbation(self, x, y, eps_step, cost_matrix):
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: Perturbations.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
            grad = np.sign(grad)
            x_adv = x + eps_step * grad

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            x_adv = x + eps_step * grad

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            x_adv = x + eps_step * grad

        elif self.norm == 'wasserstein':
            x_adv = self._conjugate_sinkhorn_optimizer()

        else:
            raise ValueError("This norm is not supported.")

        return x_adv

    def _conjugate_sinkhorn_optimizer(self):
        """

        :return:
        """


    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value)
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important. The default value 1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(Wasserstein, self).set_params(**kwargs)


        if kernel_size % 2 != 1:
            raise ValueError("Need odd kernel size")


        return True


