from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.classifiers.pytorch import PyTorchClassifier
from art.classifiers.detector_classifier import DetectorClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')


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


class TestDetectorClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the DetectorClassifier.
    """
    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Define the internal classifier
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        classifier.fit(x_train, y_train, batch_size=100, nb_epochs=2)

        # Define the internal detector
        model = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(2304, 1))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        detector = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 1)

        # Define the detector-classifier
        cls.detector_classifier = DetectorClassifier(classifier=classifier, detector=detector)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_predict(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test predict logits
        preds = self.detector_classifier.predict(x=x_test, logits=True)
        self.assertTrue(np.array(preds.shape == (NB_TEST, 11)).all())

        # Test predict softmax
        preds = self.detector_classifier.predict(x=x_test, logits=False)
        self.assertTrue(np.sum(preds) == NB_TEST)

    def test_nb_classes(self):
        dc = self.detector_classifier
        self.assertTrue(dc.nb_classes == 11)

    def test_input_shape(self):
        dc = self.detector_classifier
        self.assertTrue(np.array(dc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Get the classifier
        dc = self.detector_classifier

        # Test logits = True and label = None
        grads = dc.class_gradient(x=x_test, logits=True, label=None)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 11, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = True and label = 5
        grads = dc.class_gradient(x=x_test, logits=True, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = True and label = 10
        grads = dc.class_gradient(x=x_test, logits=True, label=10)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = True and label = array
        label = np.random.randint(11, size=NB_TEST)
        grads = dc.class_gradient(x=x_test, logits=True, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = False and label = None
        grads = dc.class_gradient(x=x_test, logits=False, label=None)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 11, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = False and label = 5
        grads = dc.class_gradient(x=x_test, logits=False, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = False and label = 10
        grads = dc.class_gradient(x=x_test, logits=False, label=10)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test logits = False and label = array
        label = np.random.randint(11, size=NB_TEST)
        grads = dc.class_gradient(x=x_test, logits=False, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_set_learning(self):
        dc = self.detector_classifier

        self.assertTrue(dc.classifier._model.training)
        self.assertTrue(dc.detector._model.training)
        self.assertTrue(dc.learning_phase is None)

        dc.set_learning_phase(False)
        self.assertFalse(dc.classifier._model.training)
        self.assertFalse(dc.detector._model.training)
        self.assertFalse(dc.learning_phase)

        dc.set_learning_phase(True)
        self.assertTrue(dc.classifier._model.training)
        self.assertTrue(dc.detector._model.training)
        self.assertTrue(dc.learning_phase)

    def test_save(self):
        model = self.detector_classifier
        import tempfile
        import os
        t_file = tempfile.NamedTemporaryFile()
        full_path = t_file.name
        t_file.close()
        base_name = os.path.basename(full_path)
        dir_name = os.path.dirname(full_path)
        model.save(base_name, path=dir_name)

        self.assertTrue(os.path.exists(full_path + "_classifier.optimizer"))
        self.assertTrue(os.path.exists(full_path + "_classifier.model"))
        os.remove(full_path + '_classifier.optimizer')
        os.remove(full_path + '_classifier.model')

        self.assertTrue(os.path.exists(full_path + "_detector.optimizer"))
        self.assertTrue(os.path.exists(full_path + "_detector.model"))
        os.remove(full_path + '_detector.optimizer')
        os.remove(full_path + '_detector.model')


if __name__ == '__main__':
    unittest.main()
