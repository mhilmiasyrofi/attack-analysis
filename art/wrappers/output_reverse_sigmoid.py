# MIT License
#
# Copyright (C) IBM Corporation 2019
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
"""
This module implements a wrapper for the Reverse Sigmoid output perturbation.

| Paper link: https://arxiv.org/abs/1806.00054
"""
import logging

import numpy as np

from art import NUMPY_DTYPE
from art.wrappers.wrapper import ClassifierWrapper
from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class ReverseSigmoid(ClassifierWrapper, Classifier):
    """
    Implementation of a classifier wrapper for the Reverse Sigmoid output perturbation.
    """

    def __init__(self, classifier, beta=1.0, gamma=0.1):
        """
        Create a wrapper for rounded prediction output.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of smoothing.
        :type classifier: :class:`.Classifier`
                :param beta: A positive magnitude parameter
        :type beta: `float`
        :param gamma: A positive dataset and model specific convergence parameter
        :type gamma: `float`
        """
        super(ReverseSigmoid, self).__init__(classifier)
        self.beta = beta
        self.gamma = gamma

    # pylint: disable=W0221
    def predict(self, x, batch_size=128, **kwargs):
        """
        Prediction the wrapped classifier and perturb output with reverse sigmoid.

        :param x: Input data
        :type x: `np.ndarray`
        :param batch_size: Batch size
        :type batch_size: `int`
        :return: Rounded predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        predictions = self.classifier.predict(x, batch_size=batch_size)

        perturbation_r = self.beta * (sigmoid(-self.gamma * (np.log((1.00001 - predictions) / predictions))) - 0.5)
        predictions_perturbed = predictions - perturbation_r
        predictions_perturbed = np.clip(predictions_perturbed, 0.0, np.finfo(NUMPY_DTYPE).max)
        alpha = 1.0 / np.sum(predictions_perturbed, axis=-1, keepdims=True)
        return alpha * predictions_perturbed

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
        :type x: `np.ndarray`
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  One Hot Encoding format.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x`, taking an expectation
        over transformations.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-hot encoded.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x`, taking an expectation over transformations.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def save(self, filename, path=None):
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :type filename: `str`
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError


if __name__ == '__main__':
    from art.utils import load_dataset
    from art.utils_test import get_classifier_kr

    (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
    classifier = get_classifier_kr()

    rp = ReverseSigmoid(classifier=classifier, beta=1.0, gamma=0.1)
    print(rp.predict(x_test[0:1]))
    print(classifier.predict(x_test[0:1]))
