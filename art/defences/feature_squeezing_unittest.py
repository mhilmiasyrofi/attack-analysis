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

import unittest

import numpy as np
import tensorflow as tf

from art.defences.feature_squeezing import FeatureSqueezing


class TestFeatureSqueezing(unittest.TestCase):
    def test_ones(self):
        m, n = 10, 2
        x = np.ones((m, n))

        for depth in range(1,50):
            with self.subTest("bit depth = {}".format(depth)):
                preproc = FeatureSqueezing()
                squeezed_x = preproc(x, depth)
                self.assertTrue((squeezed_x == 1).all())

    def test_random(self):
        m, n = 1000, 20
        x = np.random.rand(m, n)
        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)

        preproc = FeatureSqueezing()
        squeezed_x = preproc(x, 1)
        self.assertTrue((squeezed_x[x_zero] == 0.).all())
        self.assertTrue((squeezed_x[x_one] == 1.).all())

        squeezed_x = preproc(x, 2)
        self.assertFalse(np.logical_and(0. < squeezed_x, squeezed_x < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < squeezed_x, squeezed_x < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < squeezed_x, squeezed_x < 1.).any())

    def test_tf_feature_squeezing(self):
        # With tensors
        m, n = 10, 2
        sess = tf.Session()
        x = tf.ones((m, n))
        fs = FeatureSqueezing()

        for depth in range(1, 10):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = sess.run(fs._tf_predict(x, depth))
                self.assertTrue((squeezed_x == 1).all())

        # With placeholders
        x = np.ones((m, n))

        x_op = tf.placeholder(tf.float32, shape=[None, 2])
        for depth in range(1, 10):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = sess.run(fs._tf_predict(x_op, depth), feed_dict={x_op: x})
                self.assertTrue((squeezed_x == 1).all())


if __name__ == '__main__':
    unittest.main()
