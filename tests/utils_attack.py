import pytest
import numpy as np
import logging
from art import utils
from tests import utils_test
import keras.backend as k
from tests.utils_test import ExpectedValue

logger = logging.getLogger(__name__)


def _backend_targeted_images(attack, classifier, fix_get_mnist_subset):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    targets = utils.random_targets(y_test_mnist, classifier.nb_classes())
    x_test_adv = attack.generate(x_test_mnist, y=targets)
    assert (x_test_mnist == x_test_adv).all() == False

    y_test_pred_adv = utils.get_labels_np_array(classifier.predict(x_test_adv))

    assert targets.shape == y_test_pred_adv.shape
    assert (targets == y_test_pred_adv).sum() >= (x_test_mnist.shape[0] // 2)

    utils_test.check_adverse_example_x(x_test_adv, x_test_mnist)


    y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)

    target = np.argmax(targets, axis=1)
    assert (target == y_pred_adv).any()

def _backend_norm_images(attack, classifier, mnist_dataset, expected_values):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = mnist_dataset
    x_test_adv = attack.generate(x_test_mnist)
    y_test_pred_adv = classifier.predict(x_test_adv[8:9])

    if "x_test_mean" in expected_values:
        utils_test.assert_almost_equal_mean(x_test_mnist, x_test_adv, expected_values["x_test_mean"].value, decimal=expected_values["x_test_mean"].decimals)
    if "x_test_min" in expected_values:
        utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv, expected_values["x_test_min"].value, decimal=expected_values["x_test_min"].decimals)
    if "x_test_max" in expected_values:
        utils_test.assert_almost_equal_max(x_test_mnist, x_test_adv, expected_values["x_test_max"].value, decimal=expected_values["x_test_max"].decimals)
    if "y_test_pred_adv_expected" in expected_values:
        np.testing.assert_array_almost_equal(y_test_pred_adv, expected_values["y_test_pred_adv_expected"].value, decimal=expected_values["y_test_pred_adv_expected"].decimals)





def _backend_targeted_tabular(attack, classifier, fix_get_iris, fix_mlFramework):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris

    if fix_mlFramework in ["scikitlearn"]:
        classifier.fit(x=x_test_iris, y=y_test_iris)

    targets = utils.random_targets(y_test_iris, nb_classes=3)
    x_test_adv = attack.generate(x_test_iris, **{'y': targets})

    utils_test.check_adverse_example_x(x_test_adv, x_test_iris)

    y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    target = np.argmax(targets, axis=1)
    assert (target == y_pred_adv).any()

    accuracy = np.sum(y_pred_adv == np.argmax(targets, axis=1)) / y_test_iris.shape[0]
    logger.info('Success rate of targeted boundary on Iris: %.2f%%', (accuracy * 100))

def _back_end_untargeted_images(attack, classifier, fix_get_mnist_subset, fix_mlFramework):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    x_test_adv = attack.generate(x_test_mnist)

    utils_test.check_adverse_example_x(x_test_adv, x_test_mnist)

    y_pred = np.argmax(classifier.predict(x_test_mnist), axis=1)
    y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    assert (y_pred != y_pred_adv).any()

    if fix_mlFramework in ["keras"]:
        k.clear_session()

def _backend_untargeted_tabular(attack, iris_dataset, classifier, mlFramework, clipped):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = iris_dataset

    #TODO remove that platform specific case
    if mlFramework in ["scikitlearn"]:
        classifier.fit(x=x_test_iris, y=y_test_iris)

    x_test_adv = attack.generate(x_test_iris)

    # TODO remove that platform specific case
    # if mlFramework in ["scikitlearn"]:
    #     np.testing.assert_array_almost_equal(np.abs(x_test_adv - x_test_iris), .1, decimal=5)

    utils_test.check_adverse_example_x(x_test_adv, x_test_iris)
    # utils_test.check_adverse_example_x(x_test_adv, x_test_iris, bounded=clipped)

    y_pred_test_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    y_test_true = np.argmax(y_test_iris, axis=1)

    # assert (y_test_true == y_pred_test_adv).any(), "An untargeted attack should have changed SOME predictions"
    assert(y_test_true == y_pred_test_adv).all()==False, "An untargeted attack should NOT have changed all predictions"
    accuracy = np.sum(y_pred_test_adv == y_test_true) / y_test_true.shape[0]
    logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with FGM adversarial examples: '
                                                                 '%.2f%%', (accuracy * 100))