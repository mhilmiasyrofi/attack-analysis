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
import keras.backend as k

from art.attacks.extraction.knockoff_nets import KnockoffNets
from art.utils import load_dataset, master_seed
from tests.utils_test import get_image_classifier_tf
from tests.utils_test import get_image_classifier_kr
from tests.utils_test import get_image_classifier_pt
from tests.utils_test import get_tabular_classifier_tf
from tests.utils_test import get_tabular_classifier_kr
from tests.utils_test import get_tabular_classifier_pt
from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)

try:
    # Conditional import of `torch` to avoid segmentation fault errors this framework generates at import
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.info('Could not import PyTorch in utilities.')


BATCH_SIZE = 100
NB_TRAIN = 1000
NB_EPOCHS = 10
NB_STOLEN = 1000


class TestKnockoffNets(unittest.TestCase):
    """
    A unittest class for testing the KnockoffNets attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN].astype(ART_NUMPY_DTYPE)
        cls.y_train = y_train[:NB_TRAIN].astype(ART_NUMPY_DTYPE)

    def setUp(self):
        master_seed(1234)

    def test_tensorflow_classifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        victim_tfc, sess = get_image_classifier_tf()

        # Create the thieved classifier
        thieved_tfc, _ = get_image_classifier_tf(load_init=False, sess=sess)

        # Create random attack
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        thieved_tfc = attack.extract(x=self.x_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_tfc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_keras_classifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        victim_krc = get_image_classifier_kr()

        # Create the thieved classifier
        thieved_krc = get_image_classifier_kr(load_init=False)

        # Create random attack
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        thieved_krc = attack.extract(x=self.x_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_krc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)

        # Clean-up
        k.clear_session()

    def test_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        victim_ptc = get_image_classifier_pt()

        # Create the thieved classifier
        thieved_ptc = get_image_classifier_pt(load_init=False)

        # Create random attack
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        self.x_train = np.swapaxes(self.x_train, 1, 3)
        thieved_ptc = attack.extract(x=self.x_train, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_ptc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)
        self.x_train = np.swapaxes(self.x_train, 1, 3)


class TestKnockoffNetsVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train

    def setUp(self):
        master_seed(1234)

    def test_tensorflow_iris(self):
        """
        First test for TensorFlow.
        :return:
        """
        # Get the TensorFlow classifier
        victim_tfc, sess = get_tabular_classifier_tf()

        # Create the thieved classifier
        thieved_tfc, _ = get_tabular_classifier_tf(load_init=False, sess=sess)

        # Create random attack
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        thieved_tfc = attack.extract(x=self.x_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_tfc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_tfc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_keras_iris(self):
        """
        Second test for Keras.
        :return:
        """
        # Build KerasClassifier
        victim_krc = get_tabular_classifier_kr()

        # Create the thieved classifier
        thieved_krc = get_tabular_classifier_kr(load_init=False)

        # Create random attack
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        thieved_krc = attack.extract(x=self.x_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_krc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_krc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)

        # Clean-up
        k.clear_session()

    def test_pytorch_iris(self):
        """
        Third test for Pytorch.
        :return:
        """
        # Build PyTorchClassifier
        victim_ptc = get_tabular_classifier_pt()

        # Create the thieved classifier
        thieved_ptc = get_tabular_classifier_pt(load_init=False)

        # Create random attack
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='random')
        thieved_ptc = attack.extract(x=self.x_train, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Create adaptive attack
        attack = KnockoffNets(classifier=victim_ptc, batch_size_fit=BATCH_SIZE, batch_size_query=BATCH_SIZE,
                              nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, sampling_strategy='adaptive', reward='all')
        thieved_ptc = attack.extract(x=self.x_train, y=self.y_train, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.4)


if __name__ == '__main__':
    unittest.main()
