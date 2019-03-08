from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np

from art.classifiers import ClassifierWrapper
from art.utils import load_mnist, master_seed, get_classifier_kr

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestMixinWKerasClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        cls.model_mnist, _ = get_classifier_kr()

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_shapes(self):
        x_test, y_test = self.mnist[1]
        classifier = ClassifierWrapper(self.model_mnist)

        preds = classifier.predict(self.mnist[1][0])
        self.assertTrue(preds.shape == y_test.shape)

        self.assertTrue(classifier.nb_classes == 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertTrue(class_grads.shape == tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertTrue(loss_grads.shape == x_test[:11].shape)

    def test_class_gradient(self):
        (_, _), (x_test, _) = self.mnist
        classifier = ClassifierWrapper(self.model_mnist)

        # Test all gradients label
        grads = classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = classifier.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        (_, _), (x_test, y_test) = self.mnist
        classifier = ClassifierWrapper(self.model_mnist)

        # Test gradient
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]

        classifier = ClassifierWrapper(self.model_mnist)
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(x_test, i)
            act_name = classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

    def test_save(self):
        import os

        path = 'tmp'
        filename = 'model.h5'
        classifier = ClassifierWrapper(self.model_mnist)
        classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))
