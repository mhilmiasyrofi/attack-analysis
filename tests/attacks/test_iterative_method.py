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

import numpy as np

from art.attacks import BasicIterativeMethod
from art.classifiers import KerasClassifier
from art.utils import get_labels_np_array, random_targets

from tests.utils_test import TestBase
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)


class TestIterativeAttack(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 11
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_keras_mnist(self):
        classifier = get_classifier_kr()

        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        self._test_backend_mnist(classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist,
                                 self.y_test_mnist)

    def test_tensorflow_mnist(self):
        classifier, sess = get_classifier_tf()

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

        self._test_backend_mnist(classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist,
                                 self.y_test_mnist)

    def test_pytorch_mnist(self):
        classifier = get_classifier_pt()
        x_train = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)

        scores = get_labels_np_array(classifier.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(classifier.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

        self._test_backend_mnist(classifier, x_train, self.y_train_mnist, x_test, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):
        x_test_original = x_test.copy()

        # Test BIM with np.inf norm
        attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.1, batch_size=128)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('Accuracy on adversarial train examples: %.2f%%', (acc * 100))

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial test examples: %.2f%%', (acc * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def _test_mnist_targeted(self, classifier, x_test):
        x_test_original = x_test.copy()

        # Test FGSM with np.inf norm
        attack = BasicIterativeMethod(classifier, eps=1.0, eps_step=0.01, targeted=True, batch_size=128)
        # y_test_adv = to_categorical((np.argmax(y_test, axis=1) + 1)  % 10, 10)
        pred_sort = classifier.predict(x_test).argsort(axis=1)
        y_test_adv = np.zeros((x_test.shape[0], 10))
        for i in range(x_test.shape[0]):
            y_test_adv[i, pred_sort[i, -2]] = 1.0
        x_test_adv = attack.generate(x_test, y=y_test_adv)

        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertEqual(y_test_adv.shape, test_y_pred.shape)
        # This doesn't work all the time, especially with small networks
        self.assertGreaterEqual((y_test_adv == test_y_pred).sum(), x_test.shape[0] // 2)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_keras_mnist_targeted(self):
        classifier = get_classifier_kr()
        self._test_mnist_targeted(classifier, self.x_test_mnist)

    def test_tensorflow_mnist_targeted(self):
        classifier, sess = get_classifier_tf()
        self._test_mnist_targeted(classifier, self.x_test_mnist)

    def test_pytorch_mnist_targeted(self):
        classifier = get_classifier_pt()
        x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        self._test_mnist_targeted(classifier, x_test)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = BasicIterativeMethod(classifier=classifier)

        self.assertIn('For `BasicIterativeMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = BasicIterativeMethod(classifier=classifier)

        self.assertIn('For `BasicIterativeMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_clipped(self):
        classifier = get_iris_classifier_kr()

        # Test untargeted attack
        attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.1, batch_size=128)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with BIM adversarial examples: %.2f%%', (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = BasicIterativeMethod(classifier, targeted=True, eps=1, eps_step=0.1)
        x_test_adv = attack.generate(self.x_test_iris, **{'y': targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Success rate of targeted BIM on Iris: %.2f%%', (acc * 100))

    def test_keras_iris_unbounded(self):
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.2, batch_size=128)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with BIM adversarial examples: %.2f%%', (acc * 100))

    def test_tensorflow_iris(self):
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.1)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with BIM adversarial examples: %.2f%%', (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = BasicIterativeMethod(classifier, targeted=True, eps=1, eps_step=0.1)
        x_test_adv = attack.generate(self.x_test_iris, **{'y': targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Success rate of targeted BIM on Iris: %.2f%%', (acc * 100))

    def test_pytorch_iris(self):
        classifier = get_iris_classifier_pt()

        # Test untargeted attack
        attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.1)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with BIM adversarial examples: %.2f%%', (acc * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = BasicIterativeMethod(classifier, targeted=True, eps=1, eps_step=0.1, batch_size=128)
        x_test_adv = attack.generate(self.x_test_iris, **{'y': targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Success rate of targeted BIM on Iris: %.2f%%', (acc * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import SklearnClassifier

        scikitlearn_test_cases = [LogisticRegression(solver='lbfgs', multi_class='auto'),
                                  SVC(gamma='auto'),
                                  LinearSVC()]

        x_test_original = self.x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            # Test untargeted attack
            attack = BasicIterativeMethod(classifier, eps=1, eps_step=0.1)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with BIM adversarial examples: '
                                                                         '%.2f%%', (acc * 100))

            # Test targeted attack
            targets = random_targets(self.y_test_iris, nb_classes=3)
            attack = BasicIterativeMethod(classifier, targeted=True, eps=1, eps_step=0.1, batch_size=128)
            x_test_adv = attack.generate(self.x_test_iris, **{'y': targets})
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
            acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted BIM on Iris: %.2f%%',
                        (acc * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
