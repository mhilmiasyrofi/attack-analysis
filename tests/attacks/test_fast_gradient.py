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
import sys
import numpy as np
import os
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.defences import FeatureSqueezing
from art.utils import get_labels_np_array, random_targets

from tests.utils_test import TestBase
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

# tmp = os.environ
# print(os.environ["SECRET_KEY"])


class TestFastGradientMethodImages(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        tmp = os.environ["mlFramework"]
        cls.n_train = 100
        cls.n_test = 11
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]
        cls.mnist = (cls.x_train_mnist, cls.y_train_mnist), (cls.x_test_mnist, cls.y_test_mnist)
        cls.iris = (cls.x_train_iris, cls.y_train_iris), (cls.x_test_iris, cls.y_test_iris)

    @unittest.skipUnless(os.environ["mlFramework"] == "keras", "Not a Keras Method hence Skipping this test")
    def test_images_keras(self):
        classifier = get_classifier_kr()

        # Get the ready-trained Keras model
        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        defended_classifier = KerasClassifier(model=classifier._model, clip_values=(0, 1), defences=fs)

        self._test_backend_mnist(self.mnist, classifier, defended_classifier)

    @unittest.skipUnless(os.environ["mlFramework"] == "tensorflow", "Not a Tensorflow Method hence Skipping this test")
    def test_images_tensorflow(self):
        classifier, sess = get_classifier_tf()
        self._test_backend_mnist(self.mnist, classifier)

    @unittest.skipUnless(os.environ["mlFramework"] == "pytorch", "Not a pyTorch Method hence Skipping this test")
    def test_images_pytorch(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)
        test_mnist = (x_train, y_train), (x_test, y_test)

        classifier = get_classifier_pt()
        self._test_backend_mnist(test_mnist, classifier)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = FastGradientMethod(classifier=classifier)

        self.assertIn('For `FastGradientMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = FastGradientMethod(classifier=classifier)

        self.assertIn('For `FastGradientMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    @unittest.skipUnless(os.environ["mlFramework"] == "keras", "Not a Keras Method hence Skipping this test")
    def test_tabular_keras(self):
        (_, _), (x_test, y_test) = self.iris
        classifier_clipped = get_iris_classifier_kr()
        classifier_no_clip_values = KerasClassifier(model=classifier_clipped._model, use_logits=False, channel_index=1)

        self._test_backend_iris(x_test, y_test, classifier_clipped, classifier_no_clip_values)

    @unittest.skipUnless(os.environ["mlFramework"] == "tensorflow", "Not a Tensorflow Method hence Skipping this test")
    def test_tabular_tensorflow(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()
        self._test_backend_iris(x_test, y_test, classifier, batch_size=128)

    @unittest.skipUnless(os.environ["mlFramework"] == "pytorch", "Not a pyTorch Method hence Skipping this test")
    def test_tabular_pytorch(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()

        self._test_backend_iris(x_test, y_test, classifier, batch_size=128)

    def _test_backend_iris(self, x_test, y_test, classifier, classifier_no_clip_values=None, batch_size=1):
        # Test untargeted attack
        attack = FastGradientMethod(classifier, eps=.1)
        x_test_adv = attack.generate(x_test)

        self._check_x_adv(x_test_adv, x_test)


        y_pred_test_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        y_test_true = np.argmax(y_test, axis=1)

        self.assertTrue((y_test_true == y_pred_test_adv).any(),
                        "An untargeted attack should have changed SOME predictions")
        self.assertFalse((y_test_true == y_pred_test_adv).all(),
                         "An untargeted attack should NOT have changed all predictions")
        accuracy = np.sum(y_pred_test_adv == y_test_true) / y_test_true.shape[0]
        logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(y_test, nb_classes=3)
        y_targeted = np.argmax(targets, axis=1)
        attack = FastGradientMethod(classifier, targeted=True, eps=.1, batch_size=batch_size)
        x_test_adv = attack.generate(x_test, **{'y': targets})

        self._check_x_adv(x_test_adv, x_test)

        y_pred_test_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((y_targeted == y_pred_test_adv).any())
        accuracy = np.sum(y_pred_test_adv == y_targeted) / y_test_true.shape[0]
        logger.info('Success rate of targeted FGM on Iris: %.2f%%', (accuracy * 100))

        # Recreate a classifier without clip values
        if classifier_no_clip_values is not None:
            attack = FastGradientMethod(classifier_no_clip_values, eps=1)

            x_test_adv = attack.generate(x_test)

            self._check_x_adv(x_test_adv, x_test, bounded=False)

            y_test_true = np.argmax(y_test, axis=1)
            y_pred_test_adv = np.argmax(classifier_no_clip_values.predict(x_test_adv), axis=1)
            self.assertFalse((y_test_true == y_pred_test_adv).all())
            accuracy = np.sum(y_pred_test_adv == y_test_true) / y_test_true.shape[0]
            logger.info('Accuracy on Iris with FGM adversarial examples: %.2f%%', (accuracy * 100))

    def _test_no_norm(self, mnist_param, classifier):
        (x_train, y_train), (x_test, y_test) = mnist_param

        # Test FGSM with np.inf norm
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        x_test_adv = attack.generate(x_test)

        self.assertAlmostEqual(float(np.mean(x_test_adv - x_test)), 0.2346725, delta=0.002)
        self.assertAlmostEqual(float(np.min(x_test_adv - x_test)), -1.0, delta=0.00001)
        self.assertAlmostEqual(float(np.max(x_test_adv - x_test)), 1.0, delta=0.00001)

        y_test_pred = classifier.predict(x_test_adv)

        y_test_expected = np.asarray([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0])
        y_test_pred_expected = np.asarray([[7.32060298e-02, 4.03153598e-02, 2.08138078e-01, 2.27986258e-02,
                                            4.08675969e-01, 1.64286494e-02, 8.81226882e-02, 2.71510370e-02,
                                            6.36906400e-02, 5.14728837e-02],
                                           [1.10022835e-01, 2.53075064e-04, 3.09050769e-01, 8.28748848e-03,
                                            4.23537999e-01, 1.58944018e-02, 3.54500744e-03, 7.03897625e-02,
                                            5.08272983e-02, 8.19133688e-03],
                                           [8.34077671e-02, 1.68634069e-04, 1.14863992e-01, 1.49999780e-03,
                                            7.81848907e-01, 2.06214096e-03, 1.57082418e-03, 7.90233351e-03,
                                            3.35145928e-03, 3.32383858e-03],
                                           [7.94695988e-02, 6.41014650e-02, 1.19662583e-01, 6.82745054e-02,
                                            5.87757975e-02, 5.54384440e-02, 4.47857119e-02, 4.73252147e-01,
                                            2.29432248e-02, 1.32965213e-02],
                                           [1.37778342e-01, 5.23229912e-02, 8.03085491e-02, 7.07063973e-02,
                                            1.13677077e-01, 7.50706568e-02, 4.73172851e-02, 3.50361735e-01,
                                            5.30573502e-02, 1.93995778e-02],
                                           [8.26486796e-02, 2.93200690e-04, 1.66191280e-01, 2.23751366e-03,
                                            7.05350637e-01, 8.26103613e-03, 3.88561003e-03, 1.66236982e-02,
                                            9.51580610e-03, 4.99255396e-03],
                                           [9.07047242e-02, 1.30164847e-01, 1.11855730e-01, 1.26194224e-01,
                                            9.42349583e-02, 7.18590096e-02, 7.08150640e-02, 2.04494953e-01,
                                            7.27845579e-02, 2.68919170e-02],
                                           [1.95148319e-01, 4.02570218e-02, 2.53095001e-01, 1.19175367e-01,
                                            7.29087070e-02, 6.70288056e-02, 3.26904431e-02, 1.72511339e-01,
                                            3.19005176e-02, 1.52844433e-02],
                                           [2.34931588e-01, 1.05211824e-01, 2.23802328e-01, 1.19600385e-01,
                                            4.32376936e-02, 4.33373451e-02, 5.49205467e-02, 1.05997942e-01,
                                            3.16798575e-02, 3.72803509e-02],
                                           [1.16207518e-01, 7.97201619e-02, 1.15341313e-01, 2.22322136e-01,
                                            6.16359413e-02, 1.39247745e-01, 5.34978770e-02, 1.17801160e-01,
                                            6.38158098e-02, 3.04102581e-02],
                                           [3.33993286e-01, 4.45333160e-02, 6.64125085e-02, 4.82672676e-02,
                                            4.61629629e-02, 7.41390288e-02, 2.49474458e-02, 3.12782317e-01,
                                            2.46306900e-02, 2.41311267e-02]])

        np.testing.assert_array_equal(np.argmax(y_test, axis=1), y_test_expected)
        np.testing.assert_array_almost_equal(y_test_pred[0:3], y_test_pred_expected[0:3], decimal=2)

    def _test_minimal_perturbations(self,  mnist_param, classifier, x_test_original):

        (x_train, y_train), (x_test, y_test) = mnist_param
        # Test minimal perturbations
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
        attack.set_params(**attack_params)

        x_test_adv_min = attack.generate(x_test)

        self.assertAlmostEqual(float(np.mean(x_test_adv_min - x_test)), 0.03896513, delta=0.01)
        self.assertAlmostEqual(float(np.min(x_test_adv_min - x_test)), -0.30000000, delta=0.00001)
        self.assertAlmostEqual(float(np.max(x_test_adv_min - x_test)), 0.30000000, delta=0.00001)

        self.assertAlmostEqual(float(np.max(x_test_original - x_test)), 0.0, delta=0.00001)

        y_test_pred = classifier.predict(x_test_adv_min)

        y_test_pred_expected = np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0])

        np.testing.assert_array_equal(np.argmax(y_test_pred, axis=1), y_test_pred_expected)

    def _test_l1_norm(self, mnist_param, classifier):
        (x_train, y_train), (x_test, y_test) = mnist_param
        attack = FastGradientMethod(classifier, eps=1, norm=1, batch_size=128)
        x_test_adv = attack.generate(x_test)

        self.assertAlmostEqual(float(np.mean(x_test_adv - x_test)), 0.00051375, delta=0.002)
        self.assertAlmostEqual(float(np.min(x_test_adv - x_test)), -0.01486498, delta=0.001)
        self.assertAlmostEqual(float(np.max(x_test_adv - x_test)), 0.014761963, delta=0.001)

        y_test_pred = classifier.predict(x_test_adv[8:9])
        y_test_pred_expected = np.asarray([[0.17114946, 0.08205127, 0.07427921, 0.03722004, 0.28262928, 0.05035441,
                                            0.05271865, 0.12600125, 0.0811625, 0.0424339]])
        np.testing.assert_array_almost_equal(y_test_pred, y_test_pred_expected, decimal=4)

    def _test_l2_norm(self, mnist_param, classifier):
        (x_train, y_train), (x_test, y_test) = mnist_param
        attack = FastGradientMethod(classifier, eps=1, norm=2, batch_size=128)
        x_test_adv = attack.generate(x_test)

        self.assertAlmostEqual(float(np.mean(x_test_adv - x_test)), 0.007636424, delta=0.002)
        self.assertAlmostEqual(float(np.min(x_test_adv - x_test)), -0.211054801, delta=0.001)
        self.assertAlmostEqual(float(np.max(x_test_adv - x_test)), 0.209592223, delta=0.001)

        y_test_pred = classifier.predict(x_test_adv[8:9])
        y_test_pred_expected = np.asarray([[0.19395831, 0.11625732, 0.08293699, 0.04129186, 0.17826456, 0.06290703,
                                            0.06270657, 0.14066935, 0.07419015, 0.04681788]])
        np.testing.assert_array_almost_equal(y_test_pred, y_test_pred_expected, decimal=2)

    def _test_random_initialisation(self, mnist_param, classifier):
        (x_train, y_train), (x_test, y_test) = mnist_param
        attack = FastGradientMethod(classifier, num_random_init=3)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

    def _test_defended_classifier(self, mnist_param, classifier, defended_classifier=None):
        (x_train, y_train), (x_test, y_test) = mnist_param

        if defended_classifier is not None:
            attack = FastGradientMethod(defended_classifier, eps=1, batch_size=128)

            x_train_adv = attack.generate(x_train)
            self._check_x_adv(x_train_adv, x_train)
            y_train_pred_adv = get_labels_np_array(defended_classifier.predict(x_train_adv))
            y_train_labels = get_labels_np_array(y_train)
            # TODO Shouldn't the y_adv and y_expected labels be the same for the defence to be correct?
            self._check_y_pred_adv(y_train_pred_adv, y_train_labels)

            x_test_adv = attack.generate(x_test)
            self._check_x_adv(x_test_adv, x_test)
            y_test_pred_adv = get_labels_np_array(defended_classifier.predict(x_test_adv))
            self._check_y_pred_adv(y_test_pred_adv, y_test)

    def _test_backend_mnist(self, mnist_param, classifier, defended_classifier=None):
        (x_train, y_train), (x_test, y_test) = mnist_param

        x_test_original = x_test.copy()

        self._test_no_norm(mnist_param, classifier)

        self._test_minimal_perturbations(mnist_param, classifier, x_test_original)

        self._test_l1_norm(mnist_param, classifier)

        self._test_l2_norm(mnist_param, classifier)

        self._test_random_initialisation(mnist_param, classifier)

        self._test_mnist_targeted(classifier, x_test, y_test)

        self._test_defended_classifier(mnist_param, classifier, defended_classifier)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)


    def _test_mnist_targeted(self, classifier, x_test, y_test):
        # Test FGSM with np.inf norm
        attack = FastGradientMethod(classifier, eps=1.0, targeted=True)

        y_test_pred_sort = classifier.predict(x_test).argsort(axis=1)
        targets = np.zeros((x_test.shape[0], 10))
        for i in range(x_test.shape[0]):
            targets[i, y_test_pred_sort[i, -2]] = 1.0

        attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test, y=targets)
        self.assertFalse((x_test == x_test_adv).all())

        y_test_pred_adv = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertEqual(targets.shape, y_test_pred_adv.shape)
        self.assertGreaterEqual((targets == y_test_pred_adv).sum(), x_test.shape[0] // 2)

    @unittest.skipUnless(os.environ["mlFramework"] == "scikitlearn", "Not a scikitlearn method hence Skipping this test")
    def test_tabular_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import SklearnClassifier

        scikitlearn_test_cases = [LogisticRegression(solver='lbfgs', multi_class='auto'),
                                  SVC(gamma='auto'),
                                  LinearSVC()]

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            # Test untargeted attack
            eps = 0.1
            attack = FastGradientMethod(classifier, eps=eps)
            x_test_adv = attack.generate(self.x_test_iris)
            np.testing.assert_array_almost_equal(np.abs(x_test_adv - self.x_test_iris), eps, decimal=5)
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with FGM adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(self.y_test_iris, nb_classes=3)
            attack = FastGradientMethod(classifier, targeted=True, eps=0.1, batch_size=128)
            x_test_adv = attack.generate(self.x_test_iris, **{'y': targets})
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted FGM on Iris: %.2f%%',
                        (accuracy * 100))

    def _check_x_adv(self, x_adv, x_original, max=1.0, min=0.0, bounded=True):
        '''
        Performs basic checks on generated adversarial inputs (whether x_test or x_train)
        :param x_adv:
        :param x_original:
        :param max:
        :param min:
        :param bounded:
        :return:
        '''
        self.assertFalse((x_original == x_adv).all(), "x_test_adv should have been different from x_test")

        if bounded:
            self.assertLessEqual(np.amax(x_adv), max, "x_test_adv values should have all been below {0}".format(max))
            self.assertGreaterEqual(np.amin(x_adv), min, "x_test_adv values should have all been above {0}".format(min))
        else:
            self.assertTrue((x_adv > max).any(), "some x_test_adv values should been above 1".format(max))
            self.assertTrue((x_adv < min).any(), " some x_test_adv values should have all been below {0}".format(min))

    def _check_y_pred_adv(self, y_pred_adv, y_expected):
        self.assertFalse((y_expected == y_pred_adv).all())

if __name__ == '__main__':
    print(len(sys.argv))
    param1 = sys.argv.pop()
    unittest.main()
