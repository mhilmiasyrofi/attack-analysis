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
This module implements clean-label attacks on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, Union, List, TypeVar, Optional

import numpy as np

from art.attacks.attack import PoisoningAttackTransformer
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierNeuralNetwork
from art.estimators.classification.keras import KerasClassifier

logger = logging.getLogger(__name__)


class PoisoningAttackAdversarialEmbedding(PoisoningAttackTransformer):
    """
    Implementation of Adversarial Embedding attack by Tan, Shokri (2019).
    "Bypassing Backdoor Detection Algorithms in Deep Learning"

    This attack trains a classifier with an additional

    | Paper link: https://arxiv.org/abs/1905.13409
    """

    attack_params = PoisoningAttackTransformer.attack_params + PoisoningAttackBackdoor.attack_params + [
        "backdoor",
        "feature_layer",
        "target",
        "pp_poison",
        "discriminator_layer_1",
        "discriminator_layer_2",
        "regularization",
        "learning_rate",
    ]

    # Currently only supporting Keras
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)
    ClassifierType = TypeVar('ClassifierType', bound=ClassifierNeuralNetwork)

    def __init__(
            self,
            classifier: ClassifierType,
            backdoor: PoisoningAttackBackdoor,
            feature_layer: Union[int, str],
            target: Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]],
            pp_poison: Union[float, List[float]] = 0.05,
            discriminator_layer_1: int = 256,
            discriminator_layer_2: int = 128,
            regularization: float = 10,
            learning_rate: float = 1e-4,
    ):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: A neural network classifier.
        :param backdoor: The backdoor attack used to poison samples
        :param feature_layer: The layer of the original network to extract features from
        :param target: The target label to poison
        :param pp_poison: The percentage of training data to poison
        :param discriminator_layer_1: The size of the first discriminator layer
        :param discriminator_layer_2: The size of the second discriminator layer
        :param regularization: The regularization constant for the backdoor recognition part of the loss function
        :param learning_rate: The learning rate of clean-label attack optimization.
        """
        super().__init__(classifier=classifier)
        self.backdoor = backdoor
        self.feature_layer = feature_layer
        self.target = target
        self.pp_poison = pp_poison
        self.discriminator_layer_1 = discriminator_layer_1
        self.discriminator_layer_2 = discriminator_layer_2
        self.regularization = regularization
        self.train_data: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.is_backdoor: Optional[np.ndarray] = None
        self.learning_rate = learning_rate

        if isinstance(self.estimator, KerasClassifier):
            from keras import Model
            from keras.models import clone_model
            from keras.layers import GaussianNoise, Dense, BatchNormalization, LeakyReLU
            from keras.optimizers import Adam

            self.orig_model = clone_model(self.estimator.model, input_tensors=self.estimator.model.inputs)
            model_input = self.orig_model.input
            init_model_output = self.orig_model(model_input)

            # Extracting feature tensor
            if type(self.feature_layer) is int:
                feature_layer_tensor = self.orig_model.layers[self.feature_layer].output
            else:
                feature_layer_tensor = self.orig_model.get_layer(name=feature_layer).output
            feature_layer_output = Model(input=model_input, output=feature_layer_tensor)

            # Architecture for discriminator
            discriminator_input = feature_layer_output(model_input)
            discriminator_input = GaussianNoise(stddev=1)(discriminator_input)
            dense_layer_1 = Dense(self.discriminator_layer_1)(discriminator_input)
            norm_1_layer = BatchNormalization()(dense_layer_1)
            leaky_layer_1 = LeakyReLU(alpha=0.2)(norm_1_layer)
            dense_layer_2 = Dense(self.discriminator_layer_2)(leaky_layer_1)
            norm_2_layer = BatchNormalization()(dense_layer_2)
            leaky_layer_2 = LeakyReLU(alpha=0.2)(norm_2_layer)
            backdoor_detect = Dense(2, activation='softmax', name='backdoor_detect')(leaky_layer_2)

            # Creating embedded model
            self.embed_model = Model(inputs=self.orig_model.inputs, outputs=[init_model_output, backdoor_detect])

            # Add backdoor detectino loss
            model_name = self.orig_model.name
            model_loss = self.estimator.model.loss
            loss_name = 'backdoor_detect'
            loss_type = 'binary_crossentropy'
            if type(model_loss) is str:
                losses = {model_name: model_loss, loss_name: loss_type}
                loss_weights = {model_name: 1.0, loss_name: -self.regularization}
            elif type(model_loss) is dict:
                losses = model_loss
                losses[loss_name] = loss_type
                loss_weights = self.orig_model.loss_weights
                loss_weights[loss_name] = -self.regularization
            else:
                raise TypeError("Cannot read model loss value of type {}".format(type(model_loss)))

            opt = Adam(lr=self.learning_rate)
            self.embed_model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
        else:
            raise NotImplementedError

    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and target labels y

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param broadcast: whether or not to brodcast single target label
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        return self.backdoor.poison(x, y, broadcast=broadcast)

    def poison_estimator(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, nb_epochs: int = 10,
                         **kwargs) -> ClassifierType:
        """
        Train a poisoned model and return it
        :param x: Training data
        :param y: Training labels
        :param batch_size: The size of the batches used for training
        :param nb_epochs: The number of epochs to train for
        :return: A classifier with embedded backdoors
        """
        train_data = np.copy(x)
        train_labels = np.copy(y)

        # Select indices to poison
        selected_indices = np.zeros(len(x)).astype(bool)

        if type(self.pp_poison) is float:
            if type(self.target) is np.ndarray:
                not_target = np.logical_not(np.all(y == self.target, axis=1))
                selected_indices[not_target] = np.random.uniform(size=sum(not_target)) < self.pp_poison
            else:
                for src, _ in self.target:
                    all_src = np.all(y == src, axis=1)
                    selected_indices[all_src] = np.random.uniform(size=sum(all_src)) < self.pp_poison
        else:
            for pp, (src, _) in zip(self.pp_poison, self.target):
                all_src = np.all(y == src, axis=1)
                selected_indices[all_src] = np.random.uniform(size=sum(all_src)) < pp

        # Poison selected indices
        if type(self.target) is np.ndarray:
            to_be_poisoned = train_data[selected_indices]
            poison_data, poison_labels = self.poison(to_be_poisoned, y=self.target, broadcast=True)

            poison_idxs = np.arange(len(x))[selected_indices]
            for i, idx in enumerate(poison_idxs):
                train_data[idx] = poison_data[i]
                train_labels[idx] = poison_labels[i]
        else:
            for src, tgt in self.target:
                poison_mask = np.logical_and(selected_indices, np.all(y == src, axis=1))
                to_be_poisoned = train_data[poison_mask]
                src_poison_data, src_poison_labels = self.poison(to_be_poisoned, y=tgt.squeeze(axis=0), broadcast=True)
                train_data[poison_mask] = src_poison_data
                train_labels[poison_mask] = src_poison_labels

        # label 1 if is backdoor 0 otherwise
        is_backdoor = selected_indices.astype(int)

        # convert to one-hot
        is_backdoor = np.fromfunction(lambda b_idx: np.eye(2)[is_backdoor[b_idx]], shape=(len(x),), dtype=int)

        # Save current training data
        self.train_data = train_data
        self.train_labels = train_labels
        self.is_backdoor = is_backdoor

        if isinstance(self.estimator, KerasClassifier):
            # Call fit with both y and is_backdoor labels
            self.embed_model.fit(train_data, y=[train_labels, is_backdoor], batch_size=batch_size, epochs=nb_epochs,
                                 **kwargs)
            return KerasClassifier(self.orig_model)  # TODO: add other classifier params
        else:
            raise NotImplementedError("Currently only Keras is supported")

    def get_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns the training data generated from the last call to fit
        :return:
        """
        if self.train_data is not None:
            return self.train_data, self.train_labels, self.is_backdoor
        else:
            return None

    def _check_params(self) -> None:
        if type(self.feature_layer) is str:
            layer_names = {l.name for l in self.estimator.model.layers}
            if self.feature_layer not in layer_names:
                raise ValueError("Layer {} not found in model".format(self.feature_layer))
        elif type(self.feature_layer) is int:
            num_layers = len(self.estimator.model.layers)
            if abs(self.feature_layer) >= num_layers:
                raise ValueError("Feature layer {} is out of range. Network only has {} layers".format(
                    self.feature_layer, num_layers))

        if type(self.target) is np.ndarray:
            self._check_valid_label_shape(self.target)
        else:
            for source, target in self.target:
                self._check_valid_label_shape(source)
                self._check_valid_label_shape(target)

        if type(self.pp_poison) is float:
            _check_pp_poison(self.pp_poison)
        else:
            if type(self.target) is not list:
                raise ValueError("Target should be list of source label pairs")
            if len(self.pp_poison) != len(self.target):
                raise ValueError("pp_poison and target lists should be the same length")
            for pp in self.pp_poison:
                _check_pp_poison(pp)

        if self.regularization <= 0:
            raise ValueError("Regularization constant must be positive")

    def _check_valid_label_shape(self, label: np.ndarray) -> None:
        if label.shape != self.estimator.output_shape[1:]:
            raise ValueError("Invalid shape for target array. Should be {} received {}".format(
                self.estimator.output_shape[1:], label.shape))


def _check_pp_poison(pp_poison) -> None:
    """
    Return an error when a poison value is invalid
    """
    if 1 < pp_poison < 0:
        raise ValueError("pp_poison must be between 0 and 1")