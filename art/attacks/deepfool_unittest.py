from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks.deepfool import DeepFool
from art.utils import load_mnist, get_labels_np_array, master_seed
from art.utils import get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


class TestDeepFool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Keras classifier
        cls.classifier_k, sess = get_classifier_kr()

        scores = cls.classifier_k._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = cls.classifier_k._model.evaluate(x_test, y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        cls.classifier_tf, sess = get_classifier_tf()

        scores = get_labels_np_array(cls.classifier_tf.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(cls.classifier_tf.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

        # Create basic PyTorch model
        cls.classifier_py = get_classifier_pt()
        x_train, x_test = np.swapaxes(x_train, 1, 3), np.swapaxes(x_test, 1, 3)

        scores = get_labels_np_array(cls.classifier_py.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(cls.classifier_py.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @classmethod
    def tearDownClass(cls):
        k.clear_session()
        cls.classifier_tf._sess.close()
        tf.reset_default_graph()

    def test_mnist(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf,
                    'pytorch': self.classifier_py}

        for _, classifier in backends.items():
            if _ == 'pytorch':
                self._swap_axes()
            self._test_backend_mnist(classifier)
            if _ == 'pytorch':
                self._swap_axes()

    def _swap_axes(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        self.mnist = (x_train, y_train), (x_test, y_test)

    def _test_backend_mnist(self, classifier):
        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Test DeepFool
        attack = DeepFool(classifier, max_iter=5)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test, batch_size=11)

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

    def test_partial_grads(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        attack = DeepFool(self.classifier_k, max_iter=2, nb_grads=3)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(self.classifier_k.predict(x_test_adv))
        self.assertFalse((y_test == test_y_pred).all())

        acc = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial test examples: %.2f%%', (acc * 100))


if __name__ == '__main__':
    unittest.main()
