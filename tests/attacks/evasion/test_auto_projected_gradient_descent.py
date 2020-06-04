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
import logging
import pytest

import numpy as np

from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import PyTorchClassifier

from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.mark.only_with_platform("tensorflow")
def test_generate(is_tf_version_2, fix_get_mnist_subset, get_image_classifier_list_for_attack):

    if is_tf_version_2:
        classifier_list = get_image_classifier_list_for_attack(AutoProjectedGradientDescent)

        if classifier_list is None:
            logging.warning("Couldn't perform  this test because no classifier is defined")
            return

        for classifier in classifier_list:
            attack = AutoProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=0.3,
                eps_step=0.1,
                max_iter=2,
                targeted=False,
                nb_random_init=1,
                batch_size=32,
                loss_type="cross_entropy",
            )

            (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

            if isinstance(classifier, PyTorchClassifier):
                x_train_mnist = x_train_mnist.transpose((0, 3, 1, 2)).astype(np.float32)

            x_train_mnist_adv = attack.generate(x=x_train_mnist[0:1], y=y_train_mnist[0:1])

            assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist[0:1])) == pytest.approx(1.37e-09, 1.0e-09)


# def test_classifier_type_check_fail():
#     backend_test_classifier_type_check_fail(AutoProjectedGradientDescent,
#                                             [BaseEstimator, LossGradientsMixin, ClassifierMixin])


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
