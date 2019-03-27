# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.classifiers import KerasClassifier
from art.utils import load_mnist, preprocess
from art.poison_detection import ActivationDefence
import json


def main():
    try:
        print('See if poison model has been previously trained ')
        import pickle
        classifier = pickle.load(open('my_poison_classifier.p', 'rb'))
        print('Loaded model from pickle.... ')

        data_train = np.load('data_training.npz')
        x_train = data_train['x_train']
        y_train = data_train['y_train']
        is_poison_train = data_train['is_poison_train']
        x_poisoned_raw = data_train['x_raw']

        data_test = np.load('data_testing.npz')
        x_test = data_test['x_test']
        y_test = data_test['y_test']
        is_poison_test = data_test['is_poison_test']
        x_poisoned_raw_test = data_test['x_raw_test']

    except:
        # Read MNIST dataset (x_raw contains the original images):
        (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

        n_train = np.shape(x_raw)[0]
        num_selection = n_train
        random_selection_indices = np.random.choice(n_train, num_selection)
        x_raw = x_raw[random_selection_indices]
        y_raw = y_raw[random_selection_indices]

        # Poison training data
        perc_poison = .33
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = generate_backdoor(x_raw_test, y_raw_test,
                                                                                       perc_poison)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        # Add channel axis:
        x_test = np.expand_dims(x_test, axis=3)

        # Shuffle training data so poison is not together
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        is_poison_train = is_poison_train[shuffled_indices]

        # Save data used for training and testing split:
        np.savez('data_training.npz', x_train=x_train, y_train=y_train, is_poison_train=is_poison_train,
                 x_raw=x_poisoned_raw)
        np.savez('data_testing.npz', x_test=x_test, y_test=y_test, is_poison_test=is_poison_test,
                 x_raw_test=x_poisoned_raw_test)

        # Create Keras convolutional neural network - basic architecture from Keras examples
        # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        k.set_learning_phase(1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        classifier = KerasClassifier((min_, max_), model=model)

        classifier.fit(x_train, y_train, nb_epochs=50, batch_size=128)

        print('Saving poisoned model: ')
        pickle.dump(classifier, open('my_poison_classifier.p', 'wb'))

        # Also saving for Anu:
        file_name = 'anu_poison_mnist'
        model.save(file_name + '.hdf5')
        model_json = model.to_json()
        with open(file_name + '.json', "w") as json_file:
            json_file.write(model_json)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    # Evaluate the classifier on poisonous data
    preds = np.argmax(classifier.predict(x_test[is_poison_test]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test], axis=1)) / y_test[is_poison_test].shape[0]
    print("\nPoisonous test set accuracy (i.e. effectiveness of poison): %.2f%%" % (acc * 100))

    # Evaluate the classifier on clean data
    preds = np.argmax(classifier.predict(x_test[is_poison_test == 0]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test == 0], axis=1)) / y_test[is_poison_test == 0].shape[0]
    print("\nClean test set accuracy: %.2f%%" % (acc * 100))

    # Calling poisoning defence:
    defence = ActivationDefence(classifier, x_train, y_train)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    print(defence.get_params())
    defence.detect_poison(n_clusters=2, ndims=10, reduce="PCA")

    # Now fix the model
    x_new, y_fix = correct_poisoned_labels(x_train, y_train, is_poison_train)

    improvement = defence.relabel_poison_ground_truth(x_new, y_fix, test_set_split=0.7, tolerable_backdoor=0.001,
                                                      max_epochs=5, batch_epochs=10)

    # Evaluate the classifier on poisonous data after backdoor fix:
    preds = np.argmax(classifier.predict(x_test[is_poison_test]), axis=1)
    acc_after = np.sum(preds == np.argmax(y_test[is_poison_test], axis=1)) / y_test[is_poison_test].shape[0]
    print("\nPoisonous test set accuracy (i.e. effectiveness of poison) after backdoor fix: %.2f%%" % (acc_after * 100))

    print("\n Improvement after training: ", improvement)
    print('before: ', acc, ' after: ', acc_after)

    print("done :) ")


def generate_backdoor(x_clean, y_clean, percent_poison, backdoor_type='pattern', sources=np.arange(10),
                      targets=(np.arange(10) + 1) % 10):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """

    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def add_single_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`

    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_pattern_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def correct_poisoned_labels(x, y, is_poison):
    """
    Corrects the poisoned labels to fix the model
    :param x: training data
    :param y: poisoned labels
    :param is_poison: boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :return:
    """
    x_new = x[is_poison]
    y_poison = y[is_poison]

    y_fixed = np.zeros(len(y_poison))
    for i in range(len(y_poison)):
        label = np.argmax(y_poison[i])
        y_fixed[i] = (label - 1) % 10

    y_fixed = (y_fixed[:, None] == np.arange(y_fixed.max() + 1)).astype(int)  # Int array
    return x_new, y_fixed


if __name__ == '__main__':
    main()
