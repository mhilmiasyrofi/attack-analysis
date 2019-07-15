from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.classifiers import Classifier
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class ClassifierInstance(Classifier):
    def __init__(self, clip_values, channel_index=1):
        super(ClassifierInstance, self).__init__(clip_values=clip_values, channel_index=channel_index)

    def class_gradient(self, x, label=None):
        pass

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size):
        pass

    def loss_gradient(self, x, y):
        pass

    def predict(self, x, batch_size=128):
        pass

    def save(self, filename, path=None):
        pass

    def layer_names(self):
        pass

    def set_learning_phase(self, train):
        pass


class TestClassifier(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_preprocessing_normalisation(self):
        classifier = ClassifierInstance((0, 1))

        x = np.random.rand(100, 200)
        new_x = classifier._apply_preprocessing_normalization(x)
        self.assertEqual(np.sum(x - new_x), 0)

    def test_repr(self):
        classifier = ClassifierInstance((0, 1))

        repr_ = repr(classifier)
        self.assertIn('ClassifierInstance', repr_)
        self.assertIn('channel_index=1, clip_values=(0, 1)', repr_)
        self.assertIn('defences=None, preprocessing=(0, 1)', repr_)
