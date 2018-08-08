from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist


NB_TRAIN = 1000
NB_TEST = 20


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)

        return logit_output


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        result = x.view(n, -1)

        return result


class TestPyTorchClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the PyTorch-based classifier.
    """
    def _model_setup_module(self):
        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        return model, loss_fn, optimizer

    def _model_setup_sequential(self):
        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(2304, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        return model, loss_fn, optimizer

    def test_fit_predict(self):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Test fit and predict
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)

        ptc.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        preds = ptc.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Start to test
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        self.assertTrue(ptc.nb_classes == 10)

    def test_input_shape(self):
        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Start to test
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        self.assertTrue(np.array(ptc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Test gradient
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        grads = ptc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_class_gradient_target(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Test gradient
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        grads = ptc.class_gradient(x_test, label=3)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Create model
        model, loss_fn, optimizer = self._model_setup_module()

        # Test gradient
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        grads = ptc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_layers(self):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)
        x_train = np.swapaxes(x_train, 1, 3)

        # Create model
        model, loss_fn, optimizer = self._model_setup_sequential()

        # Test and get layers
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        ptc.fit(x_train, y_train, batch_size=100, nb_epochs=1)

        layer_names = ptc.layer_names
        self.assertTrue(layer_names == ['0_Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))', '1_ReLU()',
                                        '2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)',
                                        '3_Flatten()', '4_Linear(in_features=2304, out_features=10, bias=True)'])

        for i, name in enumerate(layer_names):
            act_i = ptc.get_activations(x_test, i)
            act_name = ptc.get_activations(x_test, name)
            self.assertTrue(np.sum(act_name-act_i) == 0)

        self.assertTrue(ptc.get_activations(x_test, 0).shape == (20, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 1).shape == (20, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 2).shape == (20, 16, 12, 12))
        self.assertTrue(ptc.get_activations(x_test, 3).shape == (20, 2304))
        self.assertTrue(ptc.get_activations(x_test, 4).shape == (20, 10))


if __name__ == '__main__':
    unittest.main()
