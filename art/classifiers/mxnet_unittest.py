from __future__ import absolute_import, division, print_function, unicode_literals

from mxnet import init, gluon
from mxnet.gluon import nn
import numpy as np
import unittest

from art.classifiers import MXClassifier
from art.utils import load_mnist

NB_TRAIN = 1000
NB_TEST = 20


class TestMXClassifier(unittest.TestCase):
    def setUp(self):

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        self._mnist = (x_train, y_train), (x_test, y_test)

        # Create a simple CNN - this one comes from the Gluon tutorial
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10)
            )
        net.initialize(init=init.Xavier())

        # Create optimizer
        self._model = net
        self._trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self._mnist

        # Fit classifier
        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        classifier.fit(x_train, y_train, batch_size=128, nb_epochs=2)

        preds = classifier.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        self.assertEqual(classifier.nb_classes, 10)

    def test_input_shape(self):
        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        self.assertEqual(classifier.input_shape, (1, 28, 28))

    # def test_class_gradient(self):
    #     # Get MNIST
    #     (_, _), (x_test, _) = self._mnist
    #
    #     # Create classifier
    #     classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
    #     grads = classifier.class_gradient(x_test)
    #
    #     self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
    #     self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self._mnist

        # Create classifier
        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_preprocessing(self):
        # Get MNIST
        (_, _), (x_test, _) = self._mnist

        # Create classifier
        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        classifier_preproc = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer, preprocessing=(1, 2))

        preds = classifier.predict((x_test - 1.) / 2)
        preds_preproc = classifier_preproc.predict(x_test)
        self.assertTrue(np.sum(preds - preds_preproc) == 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _) = self._mnist
        x_test = x_test[:NB_TEST]

        classifier = MXClassifier((0, 1), self._model, (1, 28, 28), 10, self._trainer)
        self.assertEqual(len(classifier.layer_names), 7)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(x_test, i)
            act_name = classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        self.assertTrue(classifier.get_activations(x_test, 0).shape == (NB_TEST, 6, 24, 24))
        self.assertTrue(classifier.get_activations(x_test, 4).shape == (NB_TEST, 784))


if __name__ == '__main__':
    unittest.main()
