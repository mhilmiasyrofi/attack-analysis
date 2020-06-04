# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the classifier `CatBoostARTClassifier` for CatBoost models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.estimators.estimator import BaseEstimator, DecisionTreeMixin
from art.estimators.classification.classifier import ClassifierMixin

logger = logging.getLogger(__name__)


class CatBoostARTClassifier(ClassifierMixin, DecisionTreeMixin, BaseEstimator):
    """
    Wrapper class for importing CatBoost models.
    """

    def __init__(
        self,
        model=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        clip_values=None,
        nb_features=None,
    ):
        """
        Create a `Classifier` instance from a CatBoost model.

        :param model: CatBoost model.
        :type model: `catboost.core.CatBoostClassifier`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param nb_features: Number of features.
        :type nb_features: `int`
        """
        # pylint: disable=E0611,E0401
        from catboost.core import CatBoostClassifier

        if not isinstance(model, CatBoostClassifier):
            raise TypeError("Model must be of type catboost.core.CatBoostClassifier")

        super(CatBoostARTClassifier, self).__init__(
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._model = model
        self._input_shape = (nb_features,)
        self._nb_classes = self._get_nb_classes()

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `catboost.core.CatBoostClassifier` and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        self._model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._nb_classes = self._get_nb_classes()

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Perform prediction
        predictions = self._model.predict_proba(x_preprocessed)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions

    def _get_nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        if self._model.classes_ is not None:
            return len(self._model.classes_)

        return None

    def save(self, filename, path=None):
        import pickle

        with open(filename + ".pickle", "wb") as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def get_trees(self):
        raise NotImplementedError
