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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
import numpy as np

from art.attacks import NewtonFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from art.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 100


class TestNewtonFool(unittest.TestCase):
    """
    A unittest class for testing the NewtonFool attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Attack
        nf = NewtonFool(tfc, max_iter=5, batch_size=100)
        x_test_adv = nf.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = tfc.predict(x_test)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= .9 * y_pred_adv_max).all())

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_classifier_kr()

        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Attack
        nf = NewtonFool(krc, max_iter=5, batch_size=100)
        x_test_adv = nf.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = krc.predict(x_test)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= .9 * y_pred_adv_max).all())

        # sess.close()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (_, _), (x_test, _) = self.mnist
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

        # Attack
        nf = NewtonFool(ptc, max_iter=5, batch_size=100)
        x_test_adv = nf.generate(x_test)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = ptc.predict(x_test)
        y_pred_adv = ptc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= .9 * y_pred_adv_max).all())


class TestNewtonFoolVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_kr()

        attack = NewtonFool(classifier, max_iter=5)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', (acc * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = NewtonFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', (acc * 100))

    def test_iris_tf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()

        attack = NewtonFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', (acc * 100))

    def test_iris_pt(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()

        attack = NewtonFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', (acc * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression, ScikitlearnSVC

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}  # ,
        # SVC: ScikitlearnSVC,
        # LinearSVC: ScikitlearnSVC}

        (_, _), (x_test, y_test) = self.iris

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test, y=y_test)

            attack = NewtonFool(classifier, max_iter=5, batch_size=128)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with NewtonFool adversarial examples'
                                                                         ': %.2f%%', (acc * 100))


if __name__ == '__main__':
    unittest.main()
