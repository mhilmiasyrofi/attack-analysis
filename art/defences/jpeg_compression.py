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

from io import BytesIO
import logging

import numpy as np
from PIL import Image

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class JpegCompression(Preprocessor):
    """
    Implement the jpeg compression defence approach. Some related papers: https://arxiv.org/pdf/1705.02900.pdf,
    https://arxiv.org/abs/1608.00853
    """
    params = ['quality', 'channel_index', 'clip_values', '_apply_fit', '_apply_predict']

    def __init__(self, clip_values, quality=50, channel_index=3, apply_fit=True, apply_predict=False):
        """
        Create an instance of jpeg compression.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(JpegCompression, self).__init__()
        self._is_fitted = True
        self.set_params(quality=quality, channel_index=channel_index, clip_values=clip_values, _apply_fit=apply_fit,
                        _apply_predict=apply_predict)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Apply JPEG compression to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`. `x` values are expected to be in
               the data range [0, 1].
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: compressed sample.
        :rtype: `np.ndarray`
        """
        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. JPEG compression can only be applied to data with spatial'
                             'dimensions.')

        if self.channel_index >= len(x.shape):
            raise ValueError('Channel index does not match input shape.')

        if np.min(x) < 0.0:
            raise ValueError('Negative values in input `x` detected. The JPEG compression defence requires unnormalized'
                             'input.')

        # Swap channel index
        if self.channel_index < 3 and len(x.shape) == 4:
            x_ = np.swapaxes(x, self.channel_index, 3)
        else:
            x_ = x.copy()

        # Convert into `uint8`
        if self.clip_values[1] == 1.0:
            x_ = x_ * 255
        x_ = x_.astype("uint8")

        # Convert to 'L' mode
        if x_.shape[-1] == 1:
            x_ = np.reshape(x_, x_.shape[:-1])

        # Compress one image at a time
        for i, xi in enumerate(x_):
            if len(xi.shape) == 2:
                xi = Image.fromarray(xi, mode='L')
            elif xi.shape[-1] == 3:
                xi = Image.fromarray(xi, mode='RGB')
            else:
                logger.log(level=40, msg="Currently only support `RGB` and `L` images.")
                raise NotImplementedError("Currently only support `RGB` and `L` images.")

            out = BytesIO()
            xi.save(out, format="jpeg", quality=self.quality)
            xi = Image.open(out)
            xi = np.array(xi)
            x_[i] = xi
            del out

        # Expand dim if black/white images
        if len(x_.shape) < 4:
            x_ = np.expand_dims(x_, 3)

        # Convert to old dtype
        if self.clip_values[1] == 1.0:
            x_ = x_ / 255.0
        x_ = x_.astype(NUMPY_DTYPE)

        # Swap channel index
        if self.channel_index < 3:
            x_ = np.swapaxes(x_, self.channel_index, 3)

        return x_, y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        # Save defense-specific parameters
        super(JpegCompression, self).set_params(**kwargs)

        if not isinstance(self.quality, (int, np.int)) or self.quality <= 0 or self.quality > 100:
            logger.error('Image quality must be a positive integer <= 100.')
            raise ValueError('Image quality must be a positive integer <= 100.')

        if not isinstance(self.channel_index, (int, np.int)) or self.channel_index <= 0:
            logger.error('Data channel must be a positive integer. The batch dimension is not a valid channel.')
            raise ValueError('Data channel must be a positive integer. The batch dimension is not a valid channel.')

        if len(self.clip_values) != 2:
            raise ValueError('`clip_values` should be a tuple of 2 floats or arrays containing the allowed'
                             'data range.')

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError('Invalid `clip_values`: min >= max.')

        if self.clip_values[0] != 0:
            raise ValueError('`clip_values` min value must be 0.')

        if self.clip_values[1] != 1.0 and self.clip_values[1] != 255:
            raise ValueError('`clip_values` max value must be either 1 or 255.')

        return True
