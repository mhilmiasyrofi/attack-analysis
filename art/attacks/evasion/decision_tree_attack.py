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
This module implements attacks on Decision Trees.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union

import numpy as np

from art.attacks.attack import EvasionAttack
from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.utils import check_and_transform_label_format, compute_success

logger = logging.getLogger(__name__)


class DecisionTreeAttack(EvasionAttack):
    """
    Close implementation of Papernot's attack on decision trees following Algorithm 2 and communication with the
    authors.

    | Paper link: https://arxiv.org/abs/1605.07277
    """

    attack_params = ["classifier", "offset"]

    def __init__(
        self, classifier: ScikitlearnDecisionTreeClassifier, offset: float = 0.001
    ) -> None:
        """
        :param classifier: A trained model of type scikit decision tree.
        :param offset: How much the value is pushed away from tree's threshold.
        """
        super(DecisionTreeAttack, self).__init__(classifier)
        if not isinstance(classifier, ScikitlearnDecisionTreeClassifier):
            raise TypeError("Model must be a decision tree model.")

        params = {"offset": offset}
        self.set_params(**params)

    def _df_subtree(
        self,
        position: int,
        original_class: Union[int, np.ndarray],
        target: Optional[int] = None,
    ) -> list:
        """
        Search a decision tree for a mis-classifying instance.

        :param position: An array with the original inputs to be attacked.
        :param original_class: original label for the instances we are searching mis-classification for.
        :param target: If the provided, specifies which output the leaf has to have to be accepted.
        :return: An array specifying the path to the leaf where the classification is either != original class or
                 ==target class if provided.
        """
        # base case, we're at a leaf
        if self.classifier.get_left_child(position) == self.classifier.get_right_child(
            position
        ):
            if target is None:  # untargeted case
                if self.classifier.get_classes_at_node(position) != original_class:
                    path = [position]
                else:
                    path = [-1]
            else:  # targeted case
                if self.classifier.get_classes_at_node(position) == target:
                    path = [position]
                else:
                    path = [-1]
        else:  # go deeper, depths first
            res = self._df_subtree(
                self.classifier.get_left_child(position), original_class, target
            )
            if res[0] == -1:
                # no result, try right subtree
                res = self._df_subtree(
                    self.classifier.get_right_child(position), original_class, target
                )
                if res[0] == -1:
                    # no desired result
                    path = [-1]
                else:
                    res.append(position)
                    path = res
            else:
                # done, it is returning a path
                res.append(position)
                path = res

        return path

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(
            y, self.classifier.nb_classes(), return_one_hot=False
        )
        x_adv = x.copy()

        for index in range(np.shape(x_adv)[0]):
            path = self.classifier.get_decision_path(x_adv[index])
            legitimate_class = np.argmax(
                self.classifier.predict(x_adv[index].reshape(1, -1))
            )
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position) < (len(path) - 1) or adv_path[0] == -1:
                ancestor = path[position]
                current_child = path[position + 1]
                # search in right subtree
                if current_child == self.classifier.get_left_child(ancestor):
                    if y is None:
                        adv_path = self._df_subtree(
                            self.classifier.get_right_child(ancestor), legitimate_class
                        )
                    else:
                        adv_path = self._df_subtree(
                            self.classifier.get_right_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
                else:  # search in left subtree
                    if y is None:
                        adv_path = self._df_subtree(
                            self.classifier.get_left_child(ancestor), legitimate_class
                        )
                    else:
                        adv_path = self._df_subtree(
                            self.classifier.get_left_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
                position = position - 1  # we are going the decision path upwards
            adv_path.append(ancestor)
            # we figured out which is the way to the target, now perturb
            # first one is leaf-> no threshold, cannot be perturbed
            for i in range(1, 1 + len(adv_path[1:])):
                go_for = adv_path[i - 1]
                threshold = self.classifier.get_threshold_at_node(adv_path[i])
                feature = self.classifier.get_feature_at_node(adv_path[i])
                # only perturb if the feature is actually wrong
                if x_adv[index][
                    feature
                ] > threshold and go_for == self.classifier.get_left_child(adv_path[i]):
                    x_adv[index][feature] = threshold - self.offset
                elif x_adv[index][
                    feature
                ] <= threshold and go_for == self.classifier.get_right_child(
                    adv_path[i]
                ):
                    x_adv[index][feature] = threshold + self.offset

        logger.info(
            "Success rate of decision tree attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv),
        )
        return x_adv

    def set_params(self, **kwargs) -> bool:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param offset: How much the value is pushed away from tree's threshold.
        """
        super(DecisionTreeAttack, self).set_params(**kwargs)

        if self.offset <= 0:
            raise ValueError("The offset parameter must be strictly positive.")

        return True
