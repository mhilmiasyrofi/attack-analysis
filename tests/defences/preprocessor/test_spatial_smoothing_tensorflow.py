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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from art.defences.preprocessor.spatial_smoothing_tensorflow import SpatialSmoothingTensorFlowV2
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


# TODO these fixtures are duplicates from test_spatial_smoothing needs refactoring
@pytest.fixture
def image_batch(channels_first):
    """
    Image fixture of shape NHWC and NCHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((2, 3, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def video_batch(channels_first):
    """
    Video fixture of shape NFHWC and NCFHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((1, 3, 2, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 4, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def tabular_batch():
    """
    Create tabular data fixture of shape (batch_size, features).
    """
    return np.zeros((2, 4))


@pytest.mark.xfail()
@pytest.mark.only_with_platform("tensorflow")
def test_spatial_smoothing_median_filter_call(art_warning, is_tf_version_2):
    try:
        if is_tf_version_2:
            test_input = np.array([[[[1], [2]], [[3], [4]]]])
            test_output = np.array([[[[1], [2]], [[3], [3]]]])
            spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False, window_size=2)

            assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
def test_spatial_smoothing_median_filter_call_expected_behavior(art_warning, is_tf_version_2):
    try:
        if is_tf_version_2:
            test_input = np.array([[[[1], [2]], [[3], [4]]]])
            test_output = np.array([[[[2], [2]], [[2], [2]]]])
            print(test_input.shape)
            spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False, window_size=2)

            assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize(
    "window_size",
    [
        1,
        2,
        pytest.param(
            10,
            marks=pytest.mark.xfail(
                reason="Window size of 10 fails, because TensorFlow requires that Padding size should be less than "
                "the corresponding input dimension."
            ),
        ),
    ],
)
@pytest.mark.only_with_platform("tensorflow")
def test_spatial_smoothing_image_data(art_warning, image_batch, channels_first, window_size, is_tf_version_2):
    try:
        if is_tf_version_2:
            test_input, test_output = image_batch

            if channels_first:
                exc_msg = "Only channels last input data is supported"
                with pytest.raises(ValueError, match=exc_msg):
                    _ = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=window_size)
            else:
                spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=window_size)
                assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
@pytest.mark.parametrize("channels_first", [True, False])
def test_spatial_smoothing_video_data(art_warning, video_batch, channels_first, is_tf_version_2):
    try:
        if is_tf_version_2:
            test_input, test_output = video_batch

            if channels_first:
                exc_msg = "Only channels last input data is supported"
                with pytest.raises(ValueError, match=exc_msg):
                    _ = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=2)
            else:
                spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=channels_first, window_size=2)
                assert_array_equal(spatial_smoothing(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
def test_non_spatial_data_error(art_warning, tabular_batch, is_tf_version_2):
    try:
        if is_tf_version_2:
            test_input = tabular_batch
            spatial_smoothing = SpatialSmoothingTensorFlowV2(channels_first=False)

            exc_msg = "Unrecognized input dimension. Spatial smoothing can only be applied to image"
            with pytest.raises(ValueError, match=exc_msg):
                spatial_smoothing(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
def test_window_size_error(art_warning, is_tf_version_2):
    try:
        if is_tf_version_2:
            exc_msg = "Sliding window size must be a positive integer."
            with pytest.raises(ValueError, match=exc_msg):
                SpatialSmoothingTensorFlowV2(window_size=0)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
def test_triple_clip_values_error(art_warning, is_tf_version_2):
    try:
        if is_tf_version_2:
            exc_msg = "'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range."
            with pytest.raises(ValueError, match=exc_msg):
                SpatialSmoothingTensorFlowV2(clip_values=(0, 1, 2))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow")
def test_relation_clip_values_error(art_warning, is_tf_version_2):
    try:
        if is_tf_version_2:
            exc_msg = "Invalid 'clip_values': min >= max."
            with pytest.raises(ValueError, match=exc_msg):
                SpatialSmoothingTensorFlowV2(clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)
