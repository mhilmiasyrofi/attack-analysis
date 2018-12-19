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

import keras
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.attacks.newtonfool import NewtonFool
from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 100


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        import torch.nn.functional as f

        x = self.pool(f.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)

        return logit_output


class TestNewtonFool(unittest.TestCase):
    """
    A unittest class for testing the NewtonFool attack.
    """
    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Get MNIST
        (x_train, y_train), (x_test, _) = self.mnist

        # Train the classifier
        tfc = TFClassifier((0, 1), input_ph, logits, output_ph, train, loss, None, sess)
        tfc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Attack
        # import time
        nf = NewtonFool(tfc, max_iter=5)

        # print("Test Tensorflow....")
        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=1)
        self.assertFalse((x_test == x_test_adv).all())
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = tfc.predict(x_test)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Get MNIST
        (x_train, y_train), (x_test, _) = self.mnist

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)
        krc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Attack
        # import time
        nf = NewtonFool(krc, max_iter=5)

        # print("Test Keras....")
        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = krc.predict(x_test)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Get MNIST
        (x_train, y_train), (x_test, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Create simple CNN
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        ptc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=1)

        # Attack
        # import time
        nf = NewtonFool(ptc, max_iter=5)

        # print("Test Pytorch....")
        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = ptc.predict(x_test)
        y_pred_adv = ptc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())


if __name__ == '__main__':
    unittest.main()
