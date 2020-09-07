# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the abstract estimators `TensorFlowEstimator` and `TensorFlowV2Estimator` for TensorFlow models.
"""
import logging
from typing import Any, Tuple, Optional

import numpy as np

from art.utils import ART_NUMPY_DTYPE
from art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
)
from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2

logger = logging.getLogger(__name__)


class TensorFlowEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow models.
    """
    import tensorflow as tf

    def __init__(self, **kwargs) -> None:
        """
        Estimator class for TensorFlow models.
        """
        import tensorflow as tf
        self._sess: "tf.python.client.session.Session" = None
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)

    @property
    def sess(self) -> "tf.python.client.session.Session":
        """
        Get current TensorFlow session.

        :return: The current TensorFlow session.
        """
        if self._sess is not None:
            return self._sess
        else:
            raise NotImplementedError("A valid TensorFlow session is not available.")


class TensorFlowV2Estimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow v2 models.
    """

    def __init__(self, **kwargs):
        """
        Estimator class for TensorFlow v2 models.
        """
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)

    def _apply_preprocessing_defences(self, x, y, fit: bool = False) -> Tuple[Any, Any]:
        """
        Apply all preprocessing defences of the estimator on the raw inputs `x` and `y`. This function is should
        only be called from function `_apply_preprocessing`.

        The method overrides art.estimators.estimator::BaseEstimator._apply_preprocessing_defences().
        It requires all defenses to have a method `forward()`.
        It converts numpy arrays to TensorFlow tensors first, then chains a series of defenses by calling
        defence.forward() which contains TensorFlow operations. At the end, it converts TensorFlow tensors
        back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :return: Tuple of `x` and `y` after applying the defences and standardisation.
        :rtype: Format as expected by the `model`
        """
        import tensorflow as tf

        if (
            not hasattr(self, "preprocessing_defences")
            or self.preprocessing_defences is None
            or len(self.preprocessing_defences) == 0
        ):
            return x, y

        if len(self.preprocessing_defences) == 1:
            # Compatible with non-TensorFlow defences if no chaining.
            defence = self.preprocessing_defences[0]
            x, y = defence(x, y)
        else:
            # Check if all defences are implemented in TensorFlow.
            for defence in self.preprocessing_defences:
                if not isinstance(defence, PreprocessorTensorFlowV2):
                    raise NotImplementedError(f"{defence.__class__} is not TensorFlow-specific.")

            # Convert np arrays to torch tensors.
            x = tf.convert_to_tensor(x)
            if y is not None:
                y = tf.convert_to_tensor(y)

            for defence in self.preprocessing_defences:
                if fit:
                    if defence.apply_fit:
                        x, y = defence.forward(x, y)
                else:
                    if defence.apply_predict:
                        x, y = defence.forward(x, y)

            # Convert torch tensors back to np arrays.
            x = x.numpy()
            if y is not None:
                y = y.numpy()

        return x, y

    def _apply_preprocessing_defences_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass to the gradients through all preprocessing defences that have been applied to `x`
        and `y` in the forward pass. This function is should only be called from function
        `_apply_preprocessing_gradient`.

        The method overrides art.estimators.estimator::LossGradientsMixin._apply_preprocessing_defences_gradient().
        It requires all defenses to have a method estimate_forward().
        It converts numpy arrays to TensorFlow tensors first, then chains a series of defenses by calling
        defence.estimate_forward() which contains differentiable estimate of the operations. At the end,
        it converts TensorFlow tensors back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param gradients: Gradients before backward pass through preprocessing defences.
        :type gradients: Format as expected by the `model`
        :param fit: `True` if the gradients are computed during training.
        :return: Gradients after backward pass through preprocessing defences.
        :rtype: Format as expected by the `model`
        """
        import tensorflow as tf

        if (
            not hasattr(self, "preprocessing_defences")
            or self.preprocessing_defences is None
            or len(self.preprocessing_defences) == 0
        ):
            return gradients

        if len(self.preprocessing_defences) == 1:
            # Compatible with non-TensorFlow defences if no chaining.
            defence = self.preprocessing_defences[0]
            gradients = defence.estimate_gradient(x, gradients)
        else:
            # Check if all defences are implemented in TensorFlow.
            for defence in self.preprocessing_defences:
                if not isinstance(defence, PreprocessorTensorFlowV2):
                    raise NotImplementedError(f"{defence.__class__} is not TensorFlowV2-specific.")

            with tf.GradientTape() as tape:
                # Convert np arrays to TensorFlow tensors.
                x = tf.convert_to_tensor(x, dtype=ART_NUMPY_DTYPE)
                tape.watch(x)
                gradients = tf.convert_to_tensor(gradients, dtype=ART_NUMPY_DTYPE)
                x_orig = x

                for defence in self.preprocessing_defences:
                    if fit:
                        if defence.apply_fit:
                            x = defence.estimate_forward(x)
                    else:
                        if defence.apply_predict:
                            x = defence.estimate_forward(x)

            x_grad = tape.gradient(target=x, sources=x_orig, output_gradients=gradients)

            # Convert torch tensors back to np arrays.
            gradients = x_grad.numpy()
            if gradients.shape != x_orig.shape:
                raise ValueError(
                    "The input shape is {} while the gradient shape is {}".format(x.shape, gradients.shape)
                )
        return gradients
