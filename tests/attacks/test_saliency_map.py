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

import numpy as np

from art.attacks import SaliencyMapMethod
from art.estimators.classifiers import KerasClassifier
from art.utils import get_labels_np_array, to_categorical

from tests.utils import TestBase
from tests.utils import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)


class TestSaliencyMap(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 2
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_keras_mnist(self):
        x_test_original = self.x_test_mnist.copy()

        # Keras classifier
        classifier = get_classifier_kr()

        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))

        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # targeted

        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(self.x_test_mnist, y=to_categorical(targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

    def test_tensorflow_mnist(self):
        x_test_original = self.x_test_mnist.copy()

        # Create basic CNN on MNIST using TensorFlow
        classifier, sess = get_classifier_tf()

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.n_train
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_train
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # targeted
        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(self.x_test_mnist, y=to_categorical(targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

    def test_pytorch_mnist(self):
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()

        # Create basic PyTorch model
        classifier = get_classifier_pt()

        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.n_train
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('\n[PyTorch, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # targeted
        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_mnist_adv = df.generate(x_test_mnist, y=to_categorical(targets, nb_classes))

        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        self.assertFalse((0.0 == x_test_mnist_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_mnist_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # untargeted
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100)
        x_test_mnist_adv = df.generate(x_test_mnist)

        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        self.assertFalse((0.0 == x_test_mnist_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_mnist_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = SaliencyMapMethod(classifier=classifier)

        self.assertIn('For `SaliencyMapMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.estimators.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = SaliencyMapMethod(classifier=classifier)

        self.assertIn('For `SaliencyMapMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_vector_clipped(self):
        classifier = get_iris_classifier_kr()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_keras_iris_vector_unbounded(self):
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())

    def test_tensorflow_iris_vector(self):
        classifier, _ = get_iris_classifier_tf()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_pytorch_iris_vector(self):
        classifier = get_iris_classifier_pt()

        attack = SaliencyMapMethod(classifier, theta=1)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.estimators.classifiers import SklearnClassifier

        scikitlearn_test_cases = [LogisticRegression(solver='lbfgs', multi_class='auto'),
                                  SVC(gamma='auto'),
                                  LinearSVC()]

        x_test_original = self.x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            attack = SaliencyMapMethod(classifier, theta=1, batch_size=128)
            x_test_iris_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
            self.assertTrue((x_test_iris_adv <= 1).all())
            self.assertTrue((x_test_iris_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with JSMA adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
