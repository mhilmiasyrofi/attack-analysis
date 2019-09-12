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

import keras
import keras.backend as k
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.attacks import FastGradientMethod, DeepFool
from art.classifiers import TensorFlowClassifier, KerasClassifier
from art.data_generators import DataGenerator
from art.defences import AdversarialTrainer
from art.utils import load_mnist, get_labels_np_array, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11
ACCURACY_DROP = 0.0  # The unit tests are too inaccurate


@unittest.skipIf(tf.__version__[0] == '2', reason='Skip AdversarialTrainer unittests for TensorFlow v2 until Keras '
                                                  'supports it')
class TestBase(unittest.TestCase):
    mnist = None
    classifier_k = None
    classifier_tf = None

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        TestBase.mnist = ((x_train, y_train), (x_test, y_test))

        TestBase.classifier_k = TestBase._cnn_mnist_k(x_train.shape[1:])
        TestBase.classifier_k.fit(x_train, y_train, nb_epochs=2, batch_size=BATCH_SIZE)

        scores = TestBase.classifier_k._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = TestBase.classifier_k._model.evaluate(x_test, y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        TestBase.classifier_tf = TestBase._cnn_mnist_tf(x_train.shape[1:])
        TestBase.classifier_tf.fit(x_train, y_train, nb_epochs=2, batch_size=BATCH_SIZE)

        scores = get_labels_np_array(TestBase.classifier_tf.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(TestBase.classifier_tf.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @staticmethod
    def _cnn_mnist_tf(input_shape):
        labels_tf = tf.placeholder(tf.float32, [None, 10])
        inputs_tf = tf.placeholder(tf.float32, [None] + list(input_shape))

        # Define the tensorflow graph
        conv = tf.layers.conv2d(inputs_tf, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_tf))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_tf = optimizer.minimize(loss)

        TestBase.sess = tf.Session()
        TestBase.sess.run(tf.global_variables_initializer())

        classifier = TensorFlowClassifier(input_ph=inputs_tf, output=logits, loss=loss, train=train_tf,
                                          labels_ph=labels_tf, sess=TestBase.sess, clip_values=(0, 1))
        return classifier

    @staticmethod
    def _cnn_mnist_k(input_shape):
        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        return classifier


class TestAdversarialTrainer(TestBase):
    """
    Test cases for the AdversarialTrainer class.
    """

    def test_classifier_match(self):
        attack = FastGradientMethod(self.classifier_k)
        adv_trainer = AdversarialTrainer(self.classifier_k, attack)

        self.assertEqual(len(adv_trainer.attacks), 1)
        self.assertEqual(adv_trainer.attacks[0].classifier, adv_trainer.classifier)

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self.mnist

        attack = FastGradientMethod(self.classifier_k)
        x_test_adv = attack.generate(x_test)
        preds = np.argmax(self.classifier_k.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier_k, attack)
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        preds_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        acc_new = np.sum(preds_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertGreaterEqual(acc_new, acc * ACCURACY_DROP)

        logger.info('Accuracy before adversarial training: %.2f%%', (acc * 100))
        logger.info('Accuracy after adversarial training: %.2f%%', (acc_new * 100))

    def test_transfer(self):
        (x_train, y_train), (x_test, y_test) = self.mnist

        attack = DeepFool(self.classifier_tf)
        x_test_adv = attack.generate(x_test)
        preds = np.argmax(self.classifier_k.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier_k, attack)
        adv_trainer.fit(x_train, y_train, nb_epochs=2, batch_size=6)

        preds_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        acc_new = np.sum(preds_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertGreaterEqual(acc_new, acc * ACCURACY_DROP)

        logger.info('Accuracy before adversarial training: %.2f%%', (acc * 100))
        logger.info('Accuracy after adversarial training: %.2f%%', (acc_new * 100))

    def test_two_attacks(self):
        (x_train, y_train), (x_test, y_test) = self.mnist

        attack1 = FastGradientMethod(self.classifier_k)
        attack2 = DeepFool(self.classifier_tf)
        x_test_adv = attack1.generate(x_test)
        preds = np.argmax(self.classifier_k.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier_k, attacks=[attack1, attack2])
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        preds_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        acc_new = np.sum(preds_new == np.argmax(y_test, axis=1)) / NB_TEST
        # No reason to assert the newer accuracy is higher. It might go down slightly
        self.assertGreaterEqual(acc_new, acc * ACCURACY_DROP)

        logger.info('Accuracy before adversarial training: %.2f%%', (acc * 100))
        logger.info('\nAccuracy after adversarial training: %.2f%%', (acc_new * 100))

    def test_two_attacks_with_generator(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train_original = x_train.copy()

        class MyDataGenerator(DataGenerator):
            def __init__(self, x, y, size, batch_size):
                super().__init__(size=size, batch_size=batch_size)
                self.x = x
                self.y = y
                self.size = size
                self.batch_size = batch_size

            def get_batch(self):
                ids = np.random.choice(self.size, size=min(self.size, self.batch_size), replace=False)
                return self.x[ids], self.y[ids]

        generator = MyDataGenerator(x_train, y_train, x_train.shape[0], 1)

        attack1 = FastGradientMethod(self.classifier_k)
        attack2 = DeepFool(self.classifier_tf)
        x_test_adv = attack1.generate(x_test)
        preds = np.argmax(self.classifier_k.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier_k, attacks=[attack1, attack2])
        adv_trainer.fit_generator(generator, nb_epochs=5)

        preds_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        acc_new = np.sum(preds_new == np.argmax(y_test, axis=1)) / NB_TEST
        # No reason to assert the newer accuracy is higher. It might go down slightly
        self.assertGreaterEqual(acc_new, acc * ACCURACY_DROP)

        logger.info('Accuracy before adversarial training: %.2f%%', (acc * 100))
        logger.info('\nAccuracy after adversarial training: %.2f%%', (acc_new * 100))

        # Finally assert that the original training data hasn't changed:
        self.assertTrue((x_train == x_train_original).all())

    def test_targeted_attack_error(self):
        """
        Test the adversarial trainer using a targeted attack, which will currently result in a
        NotImplementError.

        :return: None
        """
        (x_train, y_train), (_, _) = self.mnist
        params = {'nb_epochs': 2, 'batch_size': BATCH_SIZE}

        classifier = self.classifier_k
        adv = FastGradientMethod(classifier, targeted=True)
        adv_trainer = AdversarialTrainer(classifier, attacks=adv)
        self.assertRaises(NotImplementedError, adv_trainer.fit, x_train, y_train, **params)


if __name__ == '__main__':
    unittest.main()
