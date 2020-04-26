# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements the universal adversarial perturbations attack `UniversalPerturbation`. This is a white-box
attack.

| Paper link: https://arxiv.org/abs/1610.08401
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import types
from typing import Any, Dict, Optional, Union

import numpy as np

from art.attacks.attack import EvasionAttack
from art.classifiers.classifier import ClassifierNeuralNetwork, ClassifierGradients
from art.exceptions import ClassifierError
from art.utils import projection

logger = logging.getLogger(__name__)


class UniversalPerturbation(EvasionAttack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a fixed perturbation to be applied to all
    future inputs. To this end, it can use any adversarial attack method.

    | Paper link: https://arxiv.org/abs/1610.08401
    """

    attacks_dict = {
        "carlini": "art.attacks.evasion.carlini.CarliniL2Method",
        "carlini_inf": "art.attacks.evasion.carlini.CarliniLInfMethod",
        "deepfool": "art.attacks.evasion.deepfool.DeepFool",
        "ead": "art.attacks.evasion.elastic_net.ElasticNet",
        "fgsm": "art.attacks.evasion.fast_gradient.FastGradientMethod",
        "bim": "art.attacks.evasion.iterative_method.BasicIterativeMethod",
        "pgd": "art.attacks.evasion.projected_gradient_descent.ProjectedGradientDescent",
        "newtonfool": "art.attacks.evasion.newtonfool.NewtonFool",
        "jsma": "art.attacks.evasion.saliency_map.SaliencyMapMethod",
        "vat": "art.attacks.evasion.virtual_adversarial.VirtualAdversarialMethod",
    }
    attack_params = EvasionAttack.attack_params + [
        "attacker",
        "attacker_params",
        "delta",
        "max_iter",
        "eps",
        "norm",
    ]

    def __init__(
        self,
        classifier: Union[ClassifierGradients, ClassifierNeuralNetwork],
        attacker: str = "deepfool",
        attacker_params: Optional[Dict[str, Any]] = None,
        delta: float = 0.2,
        max_iter: int = 20,
        eps: float = 10.0,
        norm: int = np.inf,
    ) -> None:
        """
        :param classifier: A trained classifier.
        :param attacker: Adversarial attack name. Default is 'deepfool'. Supported names: 'carlini', 'carlini_inf',
                         'deepfool', 'fgsm', 'bim', 'pgd', 'margin', 'ead', 'newtonfool', 'jsma', 'vat'.
        :param attacker_params: Parameters specific to the adversarial attack. If this parameter is not specified,
                                the default parameters of the chosen attack will be used.
        :param delta: desired accuracy
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :param eps: Attack step size (input variation).
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 2.
        """
        super(UniversalPerturbation, self).__init__(classifier)
        if not isinstance(classifier, ClassifierNeuralNetwork) or not isinstance(
            classifier, ClassifierGradients
        ):
            raise ClassifierError(
                self.__class__,
                [ClassifierNeuralNetwork, ClassifierGradients],
                classifier,
            )

        kwargs = {
            "attacker": attacker,
            "attacker_params": attacker_params,
            "delta": delta,
            "max_iter": max_iter,
            "eps": eps,
            "norm": norm,
        }
        self.set_params(**kwargs)

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        logger.info(
            "Computing universal perturbation based on %s attack.", self.attacker
        )

        # Init universal perturbation
        noise = 0
        fooling_rate = 0.0
        nb_instances = len(x)

        # Instantiate the middle attacker and get the predicted labels
        attacker = self._get_attack(self.attacker, self.attacker_params)
        pred_y = self.classifier.predict(x, batch_size=1)
        pred_y_max = np.argmax(pred_y, axis=1)

        # Start to generate the adversarial examples
        nb_iter = 0
        while fooling_rate < 1.0 - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, ex in enumerate(x[rnd_idx]):
                x_i = ex[None, ...]

                current_label = np.argmax(self.classifier.predict(x_i + noise)[0])
                original_label = np.argmax(pred_y[rnd_idx][j])

                if current_label == original_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(x_i + noise)
                    new_label = np.argmax(self.classifier.predict(adv_xi)[0])

                    # If the class has changed, update v
                    if current_label != new_label:
                        noise = adv_xi - x_i

                        # Project on L_p ball
                        noise = projection(noise, self.eps, self.norm)
            nb_iter += 1

            # Apply attack and clip
            x_adv = x + noise
            if (
                hasattr(self.classifier, "clip_values")
                and self.classifier.clip_values is not None
            ):
                clip_min, clip_max = self.classifier.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)

            # Compute the error rate
            y_adv = np.argmax(self.classifier.predict(x_adv, batch_size=1), axis=1)
            fooling_rate = np.sum(pred_y_max != y_adv) / nb_instances

        self.fooling_rate = fooling_rate
        self.converged = nb_iter < self.max_iter
        self.noise = noise
        logger.info(
            "Success rate of universal perturbation attack: %.2f%%", 100 * fooling_rate
        )

        return x_adv

    def set_params(self, **kwargs) -> bool:
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param attacker: Adversarial attack name. Default is 'deepfool'. Supported names: 'carlini', 'deepfool', 'fgsm',
                'newtonfool', 'jsma', 'vat'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: desired accuracy
        :type delta: `float`
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param norm: Order of the norm. Possible values: np.inf, 2 (default is np.inf)
        :type norm: `int`
        """
        super(UniversalPerturbation, self).set_params(**kwargs)

        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")

        return True

    def _get_attack(
        self, a_name: str, params: Optional[Dict[str, Any]] = None
    ) -> EvasionAttack:
        """
        Get an attack object from its name.

        :param a_name: Attack name.
        :param params: Attack params.
        :return: Attack object.
        :raises NotImplementedError: If the attack is not supported.
        """
        try:
            attack_class = self._get_class(self.attacks_dict[a_name])
            a_instance = attack_class(self.classifier)

            if params:
                a_instance.set_params(**params)

            return a_instance
        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name))

    @staticmethod
    def _get_class(class_name: str) -> types.ModuleType:
        """
        Get a class module from its name.

        :param class_name: Full name of a class.
        :return: The class `module`.
        """
        sub_mods = class_name.split(".")
        module_ = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        class_module = getattr(module_, sub_mods[-1])

        return class_module
