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
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.attacks.extraction.copycat_cnn import CopycatCNN
from art.classifiers import TensorFlowClassifier
from art.classifiers import KerasClassifier
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_tf
from art.utils_test import get_classifier_kr
from art.utils_test import get_classifier_pt
from art.utils_test import get_iris_classifier_tf
from art.utils_test import get_iris_classifier_kr
from art.utils_test import get_iris_classifier_pt
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)

try:
    # Conditional import of `torch` to avoid segmentation fault errors this framework generates at import
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.info('Could not import PyTorch in utilities.')

if tf.__version__[0] == '2':
    # pylint: disable=E0401
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()


BATCH_SIZE = 100
NB_TRAIN = 1000
NB_EPOCHS = 10
NB_STOLEN = 1000


class TestCopycatCNN(unittest.TestCase):
    """
    A unittest class for testing the CopycatCNN attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN].astype(NUMPY_DTYPE)
        cls.y_train = y_train[:NB_TRAIN].astype(NUMPY_DTYPE)

    def setUp(self):
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifiers
        victim_tfc, sess = get_classifier_tf()

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 4, 4)
        flattened = tf.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(flattened, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)

        # TensorFlow session and initialization
        sess.run(tf.global_variables_initializer())

        # Create the classifier
        thieved_tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph,
                                           train=train, loss=loss, learning=None, sess=sess)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_tfc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_tfc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Clean-up session
        sess.close()
        tf.reset_default_graph()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        victim_krc = get_classifier_kr()

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        loss = keras.losses.categorical_crossentropy
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        # Get classifier
        thieved_krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_krc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_krc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Clean-up
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        victim_ptc = get_classifier_pt()

        class Model(nn.Module):
            """
            Create model for pytorch.
            """

            def __init__(self):
                super(Model, self).__init__()

                self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
                self.pool = nn.MaxPool2d(4, 4)
                self.fullyconnected = nn.Linear(25, 10)

            # pylint: disable=W0221
            # disable pylint because of API requirements for function
            def forward(self, x):
                """
                Forward function to evaluate the model

                :param x: Input to the model
                :return: Prediction of the model
                """
                x = self.conv(x)
                x = torch.nn.functional.relu(x)
                x = self.pool(x)
                x = x.reshape(-1, 25)
                x = self.fullyconnected(x)
                x = torch.nn.functional.softmax(x)

                return x

        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        thieved_ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28),
                                        nb_classes=10, clip_values=(0, 1))

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_ptc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)

        self.x_train = np.swapaxes(self.x_train, 1, 3)
        thieved_ptc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train[:100]), axis=1)
        self.x_train = np.swapaxes(self.x_train, 1, 3)

        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)


class TestCopycatCNNVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train

    def setUp(self):
        master_seed(1234)

    def test_iris_tf(self):
        """
        First test for TF.
        :return:
        """
        # Get the TF classifier
        victim_tfc, sess = get_iris_classifier_tf()

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 4])
        output_ph = tf.placeholder(tf.int32, shape=[None, 3])

        # Define the tensorflow graph
        dense1 = tf.layers.dense(input_ph, 10)
        dense2 = tf.layers.dense(dense1, 10)
        logits = tf.layers.dense(dense2, 3)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        sess.run(tf.global_variables_initializer())

        # Train the classifier
        thieved_tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph,
                                           train=train, loss=loss, learning=None, sess=sess, channel_index=1)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_tfc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_tfc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Clean-up session
        sess.close()
        tf.reset_default_graph()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_kr(self):
        """
        Second test for Keras.
        :return:
        """
        # Build KerasClassifier
        victim_krc, _ = get_iris_classifier_kr()

        # Create simple CNN
        model = Sequential()
        model.add(Dense(10, input_shape=(4,), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        # Get classifier
        thieved_krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False, channel_index=1)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_krc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_krc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_krc)

        victim_preds = np.argmax(victim_krc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)

        # Clean-up
        k.clear_session()

    def test_iris_pt(self):
        """
        Third test for Pytorch.
        :return:
        """
        # Build PyTorchClassifier
        victim_ptc = get_iris_classifier_pt()

        class Model(nn.Module):
            """
            Create Iris model for PyTorch.
            """

            def __init__(self):
                super(Model, self).__init__()

                self.fully_connected1 = nn.Linear(4, 10)
                self.fully_connected2 = nn.Linear(10, 10)
                self.fully_connected3 = nn.Linear(10, 3)

            # pylint: disable=W0221
            # disable pylint because of API requirements for function
            def forward(self, x):
                x = self.fully_connected1(x)
                x = self.fully_connected2(x)
                logit_output = self.fully_connected3(x)

                return logit_output

        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Get classifier
        thieved_ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(4,),
                                        nb_classes=3, clip_values=(0, 1), channel_index=1)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_ptc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_ptc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.3)


if __name__ == '__main__':
    unittest.main()
