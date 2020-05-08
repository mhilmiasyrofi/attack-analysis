# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin

logger = logging.getLogger(__name__)


class TensorFlowV2RandomizedSmoothing(RandomizedSmoothingMixin, TensorFlowV2Classifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
        model,
        nb_classes,
        input_shape,
        loss_object=None,
        train_step=None,
        channel_index=3,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
        sample_size=32,
        scale=0.1,
        alpha=0.001,
    ):
        """
        Create a randomized smoothing classifier.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
        :type nb_classes: `int`
        :param input_shape: shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :type input_shape: `tuple`
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
            the model and computing gradients of the loss w.r.t. the input.
        :type loss_object: `tf.keras.losses`
        :param train_step: a function that applies a gradient update to the trainable variables.
        :type train_step: `function`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param sample_size: Number of samples for smoothing
        :type sample_size: `int`
        :param scale: Standard deviation of Gaussian noise added.
        :type scale: `float`
        :param alpha: The failure probability of smoothing
        :type alpha: `float`
        """
        super().__init__(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=loss_object,
            train_step=train_step,
            channel_index=channel_index,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )

    def _predict_model(self, x, batch_size=128):
        return TensorFlowV2Classifier.predict(self, x=x, batch_size=128)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import tensorflow as tf

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                inputs_t = tf.convert_to_tensor(x_preprocessed)
                tape.watch(inputs_t)

                num_noise_vectors = 32

                inputs_repeat_t = tf.repeat(inputs_t, repeats=num_noise_vectors, axis=0)

                noise = (
                    tf.random.normal(
                        shape=inputs_repeat_t.shape,
                        mean=0.0,
                        stddev=1.0,
                        dtype=inputs_repeat_t.dtype,
                        seed=None,
                        name=None,
                    )
                    * self.scale
                )

                inputs_noise_t = inputs_repeat_t + noise

                inputs_noise_t = tf.clip_by_value(
                    inputs_noise_t, clip_value_min=self.clip_values[0], clip_value_max=self.clip_values[1], name=None
                )

                model_outputs = self._model(inputs_noise_t)

                softmax = tf.nn.softmax(model_outputs, axis=1, name=None)

                average_softmax = tf.reduce_mean(
                    tf.reshape(softmax, shape=(-1, num_noise_vectors, model_outputs.shape[-1])), axis=1
                )

                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        y_true=y, y_pred=average_softmax, from_logits=False, label_smoothing=0
                    )
                )

            gradients = tape.gradient(loss, inputs_t).numpy()
        else:
            raise ValueError("Expecting eager execution.")

        # Apply preprocessing gradients
        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
