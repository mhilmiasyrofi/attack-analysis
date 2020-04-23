# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import torch
from scipy.stats import truncnorm

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format, random_sphere

logger = logging.getLogger(__name__)


class ProjectedGradientDescentPytorch(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which,
    after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted
    data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "max_iter",
        "random_eps"
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False,
        num_random_init=0,
        batch_size=1,
        random_eps=False
    ):
        """
        Create a :class:`.ProjectedGradientDescentPytorch` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this
                           method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :type random_eps: `bool`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :type num_random_init: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(ProjectedGradientDescentPytorch, self).__init__(estimator)

        kwargs = {
            "norm": norm,
            "eps": eps,
            "eps_step": eps_step,
            "max_iter": max_iter,
            "targeted": targeted,
            "num_random_init": num_random_init,
            "batch_size": batch_size,
            "random_eps": random_eps
        }
        ProjectedGradientDescentPytorch.set_params(self, **kwargs)

        if self.random_eps:
            lower, upper = 0, eps
            mu, sigma = 0, (eps / 2)
            self.norm_dist = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

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
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps
            self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            self.eps_step = ratio * self.eps

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        mask = kwargs.get("mask")
        if mask is not None:
            # Ensure the mask is broadcastable
            if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape):]:
                raise ValueError("Mask shape must be broadcastable to input shape.")

        adv_x_best = None
        rate_best = None

        for _ in range(max(1, self.num_random_init)):
            adv_x = x.astype(ART_NUMPY_DTYPE)

            # Compute perturbation with implicit batching
            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                batch = x[batch_index_1:batch_index_2]
                batch_labels = targets[batch_index_1:batch_index_2]

                mask_batch = mask
                if mask is not None:
                    # Here we need to make a distinction: if the masks are different for each input, we need to index
                    # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                    if len(mask.shape) == len(x.shape):
                        mask_batch = mask[batch_index_1:batch_index_2]

                adv_x[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, mask_batch)

            if self.num_random_init > 1:
                rate = 100 * compute_success(
                    self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size
                )
                if rate_best is None or rate > rate_best or adv_x_best is None:
                    rate_best = rate
                    adv_x_best = adv_x
            else:
                adv_x_best = adv_x

        logger.info(
            "Success rate of attack: %.2f%%",
            rate_best
            if rate_best is not None
            else 100 * compute_success(self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size),
        )

        return adv_x_best

    def _generate_batch(self, x, targets, mask):
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :type targets: `np.ndarray`
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        inputs = torch.from_numpy(x.astype(ART_NUMPY_DTYPE)).to(self.estimator.get_device)
        targets = torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)).to(self.estimator.get_device)
        adv_x = inputs

        for i_max_iter in range(self.max_iter):
            adv_x = self._compute(
                adv_x,
                inputs,
                targets,
                mask,
                self.eps,
                self.eps_step,
                self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x.cpu().detach().numpy()





    def _apply_perturbation(self, batch, perturbation, eps_step):
        batch = batch + eps_step * perturbation

        if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            batch = torch.clamp(batch, clip_min, clip_max)

        return batch

    def _compute(self, x, x_init, y, eps, eps_step, random_init):
        # TODO
        # if random_init:
        #     n = x.shape[0]
        #     m = np.prod(x.shape[1:])
        #     x_adv = x.astype(ART_NUMPY_DTYPE) + (
        #         random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
        #     )
        #
        #     if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
        #         clip_min, clip_max = self.classifier.clip_values
        #         x_adv = np.clip(x_adv, clip_min, clip_max)
        # else:
        #     x_adv = x.astype(ART_NUMPY_DTYPE)

        # Get perturbation
        perturbation = self._compute_perturbation(x, y)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)
        x_adv = x_init + perturbation

        return x_adv

    @staticmethod
    def _projection(values, eps, norm_p):
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Tensor of perturbations to clip.
        :type values: `tf.Tensor`
        :param eps: Maximum norm allowed.
        :type eps: `float`
        :param norm_p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
        :type norm_p: `int`
        :return: Values of `values` after projection.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        #tol = 10e-8
        #values_tmp = values.reshape((values.shape[0], -1))

        if norm_p == 2:
            pass
            # TODO
            # values_tmp = values_tmp * np.expand_dims(
            #     np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
            # )
        elif norm_p == 1:
            pass
            # TODO
            # values_tmp = values_tmp * np.expand_dims(
            #     np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1
            # )
        elif norm_p == np.inf:
            values = torch.clamp(values, -eps, eps)
        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.")

        # values = values_tmp.reshape(values.shape)
        return values

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :type num_random_init: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(ProjectedGradientDescentPytorch, self).set_params(**kwargs)

        if self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")

        return True
