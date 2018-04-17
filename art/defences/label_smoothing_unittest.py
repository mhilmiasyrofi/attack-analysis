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

from art.defences.label_smoothing import LabelSmoothing


class TestLabelSmoothing(unittest.TestCase):
    def test_default(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        ls = LabelSmoothing()
        _, smooth_y = ls(None, y)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m)*0.9).all())

    def test_customizing(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        ls = LabelSmoothing()
        _, smooth_y = ls(None, y, max_value=1./n)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m) / n).all())
        self.assertTrue(np.isclose(smooth_y, np.ones((m, n)) / n).all())


if __name__ == '__main__':
    unittest.main()
