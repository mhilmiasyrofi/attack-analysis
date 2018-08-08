from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import unittest
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.detection.detector import BinaryInputDetector, BinaryActivationDetector
from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.keras import KerasClassifier
from art.utils import load_mnist


class TestBinaryInputDetector(unittest.TestCase):
    """
    A unittest class for testing the binary input detector.
    """
    def test_binary_input_detector(self):
        """
        Test the binary input detector end-to-end.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        input_shape = x_train.shape[1:]
        nb_classes = 10
        
        # Create simple CNN      
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        
        # Create classifier and train it:
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)
        
        # Generate adversarial samples:
        attacker = FastGradientMethod(classifier, eps=0.1)
        x_train_adv = attacker.generate(x_train[:nb_train])
        x_test_adv = attacker.generate(x_test[:nb_test])
        
        # Compile training data for detector:
        x_train_detector = np.concatenate((x_train[:nb_train], x_train_adv), axis=0)
        y_train_detector = np.concatenate((np.array([[1,0]]*nb_train), np.array([[0,1]]*nb_train)), axis=0)
        
        # Create a simple CNN for the detector.
        # Note: we use the same architecture as for the classifier, except for the number of outputs (=2)
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        
        # Create detector and train it:
        detector = BinaryInputDetector(KerasClassifier((0, 1), model, use_logits=False))
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)
        
        # Apply detector on clean and adversarial test data:
        test_detection = np.argmax(detector(x_test), axis=1)
        test_adv_detection = np.argmax(detector(x_test_adv), axis=1)

        # Assert there is at least one true positive and negative:
        nb_true_positives = len(np.where(test_adv_detection==1)[0])
        nb_true_negatives = len(np.where(test_detection==0)[0])        
        self.assertTrue(nb_true_positives > 0)
        self.assertTrue(nb_true_negatives > 0)


class TestBinaryActivationDetector(unittest.TestCase):
    """
    A unittest class for testing the binary activation detector.
    """
    def test_binary_activation_detector(self):
        """
        Test the binary activation detector end-to-end.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        input_shape = x_train.shape[1:]
        nb_classes = 10
        
        # Create simple CNN      
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        
        # Create classifier and train it:
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        classifier.fit(x_train[:nb_train], y_train[:nb_train], nb_epochs=5, batch_size=128)
        
        # Generate adversarial samples:
        attacker = FastGradientMethod(classifier, eps=0.1)
        x_train_adv = attacker.generate(x_train[:nb_train])
        x_test_adv = attacker.generate(x_test[:nb_test])
        
        # Compile training data for detector:
        x_train_detector = np.concatenate((x_train[:nb_train], x_train_adv), axis=0)
        y_train_detector = np.concatenate((np.array([[1,0]]*nb_train), np.array([[0,1]]*nb_train)), axis=0)
        
        # Create a simple CNN for the detector.
        # Note: we use the same architecture as for the classifier, except:
        # - input layer is above the one at which we consider activation (which is layer=0)
        # - the number of outputs (=2)
        activation_shape = classifier.get_activations(x_test[:1], 0).shape[1:]
        number_outputs = 2
        model = Sequential()
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=activation_shape))
        model.add(Flatten())
        model.add(Dense(number_outputs, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        
        # Create detector and train it.
        # Detector consider activations at layer=0:
        detector = BinaryActivationDetector(classifier=classifier,
                                            detector=KerasClassifier((0, 1), model, use_logits=False), 
                                            layer=0)
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)
        
        # Apply detector on clean and adversarial test data:
        test_detection = np.argmax(detector(x_test), axis=1)
        test_adv_detection = np.argmax(detector(x_test_adv), axis=1)

        # Assert there is at least one true positive and negative:
        nb_true_positives = len(np.where(test_adv_detection==1)[0])
        nb_true_negatives = len(np.where(test_detection==0)[0])        
        self.assertTrue(nb_true_positives > 0)
        self.assertTrue(nb_true_negatives > 0)

        
if __name__ == '__main__':
    unittest.main()
