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

# import os
# import shutil
import logging
import unittest
# import pickle

import tensorflow as tf
import numpy as np

# from art.config import ART_DATA_PATH
from art.utils import load_dataset, master_seed
from tests.utils_test import get_classifier_tf
from art.data_generators import TFDataGenerator

logger = logging.getLogger(__name__)

NB_TRAIN = 1000
NB_TEST = 20


class TestTensorFlowClassifier(unittest.TestCase):
    """
    This class tests the TensorFlow classifier.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        cls.classifier, cls.sess = get_classifier_tf()
        cls.classifier_logits, _ = get_classifier_tf(from_logits=True)

        if tf.__version__[0] == '2':
            cls.is_version_2 = True
        else:
            cls.is_version_2 = False

    def setUp(self):
        master_seed(1234)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [[0.12109935, 0.0498215, 0.0993958, 0.06410097, 0.11366927, 0.04645343, 0.06419806, 0.30685693,
                       0.07616713, 0.05823758]]

        for i in range(10):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[0][i], places=4)

    def test_fit_generator(self):
        if not self.is_version_2:
            classifier, sess = get_classifier_tf()

            # Create TensorFlow data generator
            x_tensor = tf.convert_to_tensor(self.x_train.reshape(10, 100, 28, 28, 1))
            y_tensor = tf.convert_to_tensor(self.y_train.reshape(10, 100, 10))
            dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
            iterator = dataset.make_initializable_iterator()
            data_gen = TFDataGenerator(sess=sess, iterator=iterator, iterator_type='initializable', iterator_arg={},
                                       size=1000, batch_size=100)

            # Test fit and predict
            classifier.fit_generator(data_gen, nb_epochs=2)
            predictions = classifier.predict(self.x_test)
            predictions_class = np.argmax(predictions, axis=1)
            true_class = np.argmax(self.y_test, axis=1)
            accuracy = np.sum(predictions_class == true_class) / len(true_class)

            logger.info('Accuracy after fitting TensorFlow classifier with generator: %.2f%%', (accuracy * 100))
            self.assertEqual(accuracy, 0.65)

    def test_nb_classes(self):
        self.assertEqual(self.classifier.nb_classes(), 10)

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (28, 28, 1))

    def test_class_gradient(self):

        # Test all gradients label = None
        gradients = self.classifier_logits.class_gradient(self.x_test)

        self.assertEqual(gradients.shape, (NB_TEST, 10, 28, 28, 1))

        expected_gradients_1 = np.asarray([-0.03347399, -0.03195872, -0.02650188, 0.04111874, 0.08676253, 0.03339913,
                                           0.06925241, 0.09387045, 0.15184258, -0.00684002, 0.05070481, 0.01409407,
                                           -0.03632583, 0.00151133, 0.05102589, 0.00766463, -0.00898967, 0.00232938,
                                           -0.00617045, -0.00201032, 0.00410065, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.09723657, -0.00240533, 0.02445251, -0.00035474, 0.04765627, 0.04286841,
                                           0.07209076, 0.0, 0.0, -0.07938144, -0.00142567, 0.02882954,
                                           -0.00049514, 0.04170151, 0.05102589, 0.09544909, -0.04401167, -0.06158172,
                                           0.03359772, -0.00838454, 0.01722163, -0.13376027, 0.08206709, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, :, 14, 0], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = self.classifier_logits.class_gradient(self.x_test, label=5)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray([-0.03347399, -0.03195872, -0.02650188, 0.04111874, 0.08676253, 0.03339913,
                                           0.06925241, 0.09387045, 0.15184258, -0.00684002, 0.05070481, 0.01409407,
                                           -0.03632583, 0.00151133, 0.05102589, 0.00766463, -0.00898967, 0.00232938,
                                           -0.00617045, -0.00201032, 0.00410065, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.09723657, -0.00240533, 0.02445251, -0.00035474, 0.04765627, 0.04286841,
                                           0.07209076, 0.0, 0.0, -0.07938144, -0.00142567, 0.02882954,
                                           -0.00049514, 0.04170151, 0.05102589, 0.09544909, -0.04401167, -0.06158172,
                                           0.03359772, -0.00838454, 0.01722163, -0.13376027, 0.08206709, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        gradients = self.classifier_logits.class_gradient(self.x_test, label=label)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 28, 28, 1))

        expected_gradients_1 = np.asarray([0.06860766, 0.065502, 0.08539103, 0.13868105, -0.05520725, -0.18788849,
                                           0.02264893, 0.02980516, 0.2226511, 0.11288887, -0.00678776, 0.02045561,
                                           -0.03120914, 0.00642691, 0.08449504, 0.02848018, -0.03251382, 0.00854315,
                                           -0.02354656, -0.00767687, 0.01565931, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.0487146, -0.0171556, -0.03161772, -0.0420007, 0.03360246, -0.01864819,
                                           0.00315916, 0.0, 0.0, -0.07631349, -0.00374462, 0.04229517,
                                           -0.01131879, 0.05044588, 0.08449504, 0.12417868, 0.07536847, 0.03906382,
                                           0.09467953, 0.00543209, -0.00504872, -0.03366479, -0.00385999, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):

        gradients = self.classifier.loss_gradient(self.x_test, self.y_test)

        self.assertEqual(gradients.shape, (NB_TEST, 28, 28, 1))

        expected_gradients_1 = np.asarray([0.00279603, 0.00266946, 0.0032446, 0.00396258, -0.00201465, -0.00564073,
                                           0.0009253, 0.00016253, 0.0040816, 0.00166697, 0.0015883, -0.00121023,
                                           -0.00390778, -0.00234937, 0.0053558, 0.00204322, -0.00172054, 0.00053564,
                                           -0.0021146, -0.00069308, 0.00141374, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([1.05401428e-04, 1.06959546e-04, 2.60490313e-04, 2.74000311e-04,
                                           -1.15295035e-04, 2.16038228e-04, 1.37472380e-04, 0.00000000e+00,
                                           0.00000000e+00, -2.91720475e-03, -3.08302144e-04, 2.63109524e-03,
                                           -1.18699251e-03, 2.63655302e-03, 5.35579538e-03, 6.38693338e-03,
                                           3.44644510e-03, 6.68899389e-04, 5.01601025e-03, 8.40547902e-04,
                                           -1.43233046e-05, -2.79442966e-03, 7.37082795e-04, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_layers(self):
        if not self.is_version_2:
            layer_names = self.classifier.layer_names

            for i, name in enumerate(layer_names):
                activation_i = self.classifier.get_activations(self.x_test, i, batch_size=5)
                activation_name = self.classifier.get_activations(self.x_test, name, batch_size=5)
                np.testing.assert_array_equal(activation_name, activation_i)

    # Commented because of problems with multiple classifiers in the same test module
    # def test_save(self):
    #     if not self.is_version_2:
    #         path = 'tmp'
    #         filename = 'model.ckpt'
    #
    #         # Save
    #         self.classifier.save(filename, path=path)
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.data-00000-of-00001')))
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'variables/variables.index')))
    #         self.assertTrue(os.path.isfile(os.path.join(path, filename, 'saved_model.pb')))
    #
    #         # # Restore
    #         # with tf.Session(graph=tf.Graph()) as sess:
    #         #     tf.saved_model.loader.load(sess, ["serve"], os.path.join(path, filename))
    #         #     graph = tf.get_default_graph()
    #         #     sess.run('SavedOutput:0', feed_dict={'SavedInputPhD:0': input_batch})
    #
    #         # Remove saved files
    #         shutil.rmtree(os.path.join(path, filename))

    def test_set_learning(self):
        if not self.is_version_2:
            self.assertEqual(self.classifier._feed_dict, {})
            self.classifier.set_learning_phase(False)
            self.assertFalse(self.classifier._feed_dict[self.classifier._learning])
            self.classifier.set_learning_phase(True)
            self.assertTrue(self.classifier._feed_dict[self.classifier._learning])
            self.assertTrue(self.classifier.learning_phase)

    def test_repr(self):

        repr_classifier = repr(self.classifier)

        if self.is_version_2:
            self.assertIn('TensorFlowV2Classifier', repr_classifier)
            self.assertIn('model=', repr_classifier)
            self.assertIn('nb_classes=10', repr_classifier)
            self.assertIn('input_shape=(28, 28, 1)', repr_classifier)
            self.assertIn('loss_object=<tensorflow.python.keras.losses.SparseCategoricalCrossentropy', repr_classifier)
            self.assertIn('train_step=<function get_classifier_tf_v2.<locals>.train_step', repr_classifier)
        else:
            self.assertIn('TensorFlowClassifier', repr_classifier)
            self.assertIn('input_ph=<tf.Tensor \'Placeholder:0\' shape=(?, 28, 28, 1) dtype=float32>', repr_classifier)
            self.assertIn('output=<tf.Tensor \'Softmax:0\' shape=(?, 10) dtype=float32>', repr_classifier)
            self.assertIn('labels_ph=<tf.Tensor \'Placeholder_1:0\' shape=(?, 10) dtype=int32>', repr_classifier)
            self.assertIn('train=<tf.Operation \'Adam\' type=NoOp>', repr_classifier)
            self.assertIn('loss=<tf.Tensor \'Mean:0\' shape=() dtype=float32>', repr_classifier)
            self.assertIn('learning=None', repr_classifier)
            self.assertIn('sess=<tensorflow.python.client.session.Session object', repr_classifier)
            self.assertIn('TensorFlowClassifier', repr_classifier)

        self.assertIn('channel_index=3, clip_values=(0, 1), defences=None, preprocessing=(0, 1))', repr_classifier)

    # Commented because of problems with multiple classifiers in the same test module
    # def test_pickle(self):
    #     if not self.is_version_2:
    #         classifier = self.classifier
    #
    #         full_path = os.path.join(ART_DATA_PATH, 'my_classifier')
    #         folder = os.path.split(full_path)[0]
    #
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)
    #
    #         pickle.dump(classifier, open(full_path, 'wb'))
    #
    #         # Unpickle:
    #         with open(full_path, 'rb') as f:
    #             classifier_loaded = pickle.load(f)
    #             self.assertEqual(classifier._clip_values, classifier_loaded._clip_values)
    #             self.assertEqual(classifier._channel_index, classifier_loaded._channel_index)
    #             self.assertEqual(set(classifier.__dict__.keys()), set(classifier_loaded.__dict__.keys()))
    #
    #         # Test predict
    #         predictions_1 = classifier.predict(self.x_test)
    #         accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
    #         predictions_2 = classifier_loaded.predict(self.x_test)
    #         accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
    #         self.assertEqual(accuracy_1, accuracy_2)


if __name__ == '__main__':
    unittest.main()
