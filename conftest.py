import json
import logging
import pytest
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from mxnet import gluon
import os
import requests
import tempfile
from torch.utils.data import DataLoader
import torch
import shutil
from tests.utils import master_seed, get_image_classifier_kr, get_image_classifier_tf, get_image_classifier_pt
from tests.utils import get_tabular_classifier_kr, get_tabular_classifier_tf, get_tabular_classifier_pt
from tests.utils import get_tabular_classifier_scikit_list, load_dataset, get_image_classifier_kr_tf
from tests.utils import get_image_classifier_mxnet_custom_ini, get_image_classifier_kr_tf_with_wildcard
from tests.utils import get_image_classifier_kr_tf_functional, get_image_classifier_kr_functional
from art.data_generators import PyTorchDataGenerator, TensorFlowDataGenerator, KerasDataGenerator, MXDataGenerator
from art.estimators.classification import KerasClassifier

logger = logging.getLogger(__name__)
art_supported_frameworks = ["keras", "tensorflow", "pytorch", "scikitlearn", "kerastf", "mxnet"]

master_seed(1234)

default_framework = "tensorflow"


def pytest_addoption(parser):
    parser.addoption(
        "--mlFramework", action="store", default=default_framework,
        help="ART tests allow you to specify which mlFramework to use. The default mlFramework used is tensorflow. "
             "Other options available are {0}".format(art_supported_frameworks)
    )


@pytest.fixture
def get_image_classifier_list_defended(framework):
    def _get_image_classifier_list_defended(one_classifier=False, **kwargs):
        sess = None
        classifier_list = None
        from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing
        clip_values = (0, 1)
        fs = FeatureSqueezing(bit_depth=2, clip_values=clip_values)

        defenses = []
        if kwargs.get("defenses") is None:
            defenses.append(fs)
        else:
            if "FeatureSqueezing" in kwargs.get("defenses"):
                defenses.append(fs)
            if "JpegCompression" in kwargs.get("defenses"):
                defenses.append(JpegCompression(clip_values=clip_values, apply_predict=True))
            if "SpatialSmoothing" in kwargs.get("defenses"):
                defenses.append(SpatialSmoothing())
            del kwargs["defenses"]

        if framework == "keras":
            classifier = get_image_classifier_kr(**kwargs)
            # Get the ready-trained Keras model

            classifier_list = [
                KerasClassifier(model=classifier._model, clip_values=(0, 1), preprocessing_defences=defenses)]

        if framework == "tensorflow":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "pytorch":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "scikitlearn":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "kerastf":
            classifier = get_image_classifier_kr_tf(**kwargs)
            classifier_list = [
                KerasClassifier(model=classifier._model, clip_values=(0, 1), preprocessing_defences=defenses)]

        if classifier_list is None:
            return None, None

        if one_classifier:
            return classifier_list[0], sess

        return classifier_list, sess

    return _get_image_classifier_list_defended


@pytest.fixture
def get_image_classifier_list_for_attack(get_image_classifier_list, get_image_classifier_list_defended):
    def get_image_classifier_list_for_attack(attack, defended=False, **kwargs):
        if defended:
            classifier_list, _ = get_image_classifier_list_defended(kwargs)
        else:
            classifier_list, _ = get_image_classifier_list()
        if classifier_list is None:
            return None

        return [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

    return get_image_classifier_list_for_attack


@pytest.fixture(autouse=True)
def setup_tear_down_framework(framework):
    # Ran before each test
    if framework == "keras":
        pass
    if framework == "tensorflow":
        # tf.reset_default_graph()
        if tf.__version__[0] != '2':
            tf.reset_default_graph()
    if framework == "pytorch":
        pass
    if framework == "scikitlearn":
        pass
    yield True

    # Ran after each test
    if framework == "keras":
        keras.backend.clear_session()
    if framework == "tensorflow":
        pass
    if framework == "pytorch":
        pass
    if framework == "scikitlearn":
        pass


@pytest.fixture
def image_iterator(framework, is_tf_version_2, get_default_mnist_subset, default_batch_size):
    (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset

    if framework == "keras" or framework == "kerastf":
        keras_gen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            rotation_range=12,
            shear_range=0.075,
            zoom_range=0.05,
            fill_mode="constant",
            cval=0,
        )
        return keras_gen.flow(x_train_mnist, y_train_mnist, batch_size=default_batch_size)

    if framework == "tensorflow":
        if not is_tf_version_2:
            x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(10, 100, 28, 28, 1))
            y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(10, 100, 10))
            # tmp = x_train_mnist.shape[0] / default_batch_size
            # x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(tmp, default_batch_size, 28, 28, 1))
            # y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(tmp, default_batch_size, 10))
            dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
            return dataset.make_initializable_iterator()

    if framework == "pytorch":
        # Create tensors from data
        x_train_tens = torch.from_numpy(x_train_mnist)
        x_train_tens = x_train_tens.float()
        y_train_tens = torch.from_numpy(y_train_mnist)
        dataset = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
        return DataLoader(dataset=dataset, batch_size=default_batch_size, shuffle=True)

    if framework == "mxnet":
        dataset = gluon.data.dataset.ArrayDataset(x_train_mnist, y_train_mnist)
        return gluon.data.DataLoader(dataset, batch_size=5, shuffle=True)

    return None


@pytest.fixture
def image_data_generator(framework, is_tf_version_2, get_default_mnist_subset, image_iterator, default_batch_size):
    def _image_data_generator(**kwargs):
        (x_train_mnist, y_train_mnist), (_, _) = get_default_mnist_subset

        if framework == "keras" or framework == "kerastf":
            return KerasDataGenerator(
                iterator=image_iterator,
                size=x_train_mnist.shape[0],
                batch_size=default_batch_size,
            )

        if framework == "tensorflow":
            if not is_tf_version_2:
                return TensorFlowDataGenerator(
                    sess=kwargs["sess"], iterator=image_iterator, iterator_type="initializable", iterator_arg={},
                    size=x_train_mnist.shape[0],
                    batch_size=default_batch_size
                )

        if framework == "pytorch":
            return PyTorchDataGenerator(iterator=image_iterator, size=x_train_mnist.shape[0],
                                        batch_size=default_batch_size)

        if framework == "mxnet":
            return MXDataGenerator(iterator=image_iterator, size=x_train_mnist.shape[0], batch_size=default_batch_size)

    return _image_data_generator


@pytest.fixture
def store_expected_values(request, is_tf_version_2):
    '''
    Stores expected values to be retrieved by the expected_values fixture
    Note1: Numpy arrays MUST be converted to list before being stored as json
    Note2: It's possible to store both a framework independent and framework specific value. If both are stored the
    framework specific value will be used
    :param request:
    :return:
    '''

    def _store_expected_values(values_to_store, framework=""):

        framework_name = framework
        if framework == "tensorflow":
            if is_tf_version_2:
                framework_name = "tensorflow2"
            else:
                framework_name = "tensorflow1"
        if framework_name is not "":
            framework_name = "_" + framework_name

        file_name = request.node.location[0].split("/")[-1][:-3] + ".json"

        try:
            with open(os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name),
                      "r") as f:
                expected_values = json.load(f)
        except FileNotFoundError:
            expected_values = {}

        test_name = request.node.name + framework_name
        expected_values[test_name] = values_to_store

        with open(os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name),
                  "w") as f:
            json.dump(expected_values, f, indent=4)

    return _store_expected_values


@pytest.fixture
def expected_values(framework, request, is_tf_version_2):
    '''
    Retrieves the expected values that were stored using the store_expected_values fixture
    :param request:
    :return:
    '''

    file_name = request.node.location[0].split("/")[-1][:-3] + ".json"

    framework_name = framework
    if framework == "tensorflow":
        if is_tf_version_2:
            framework_name = "tensorflow2"
        else:
            framework_name = "tensorflow1"
    if framework_name is not "":
        framework_name = "_" + framework_name

    with open(os.path.join(os.path.dirname(__file__), os.path.dirname(request.node.location[0]), file_name), "r") as f:
        expected_values = json.load(f)

        # searching first for any framework specific expected value
        framework_specific_values = request.node.name + framework_name
        if framework_specific_values in expected_values:
            return expected_values[framework_specific_values]
        elif request.node.name in expected_values:
            return expected_values[request.node.name]
        else:
            raise NotImplementedError(
                "Couldn't find any expected values for test {0} and framework {1}".format(request.node.name,
                                                                                          framework_name))


@pytest.fixture(scope="session")
def get_image_classifier_mx_model():
    from mxnet.gluon import nn

    # TODO needs to be made parameterizable once Mxnet allows multiple identical models to be created in one session
    from_logits = True

    class Model(nn.Block):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)
            self.model = nn.Sequential()
            self.model.add(
                nn.Conv2D(channels=1, kernel_size=7, activation="relu", ),
                nn.MaxPool2D(pool_size=4, strides=4),
                nn.Flatten(),
                nn.Dense(10, activation=None, ),
            )

        def forward(self, x):
            y = self.model(x)
            if from_logits:
                return y

            return y.softmax()

    model = Model()
    custom_init = get_image_classifier_mxnet_custom_ini()
    model.initialize(init=custom_init)
    return model


@pytest.fixture
def get_image_classifier_mx_instance(get_image_classifier_mx_model, mnist_shape):
    import mxnet
    from art.estimators.classification import MXClassifier

    model = get_image_classifier_mx_model

    def _get_image_classifier_mx_instance(from_logits=True):
        if from_logits is False:
            # due to the fact that only 1 instance of get_image_classifier_mx_model can be created in one session
            # this will be resolved once Mxnet allows for 2 models with identical weights to be created in 1 session
            raise NotImplementedError("Currently only supporting Mxnet classifier with from_logit set to True")

        loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=from_logits)
        trainer = mxnet.gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": 0.1})

        # Get classifier
        mxc = MXClassifier(
            model=model,
            loss=loss,
            input_shape=mnist_shape,
            # input_shape=(28, 28, 1),
            nb_classes=10,
            optimizer=trainer,
            ctx=None,
            channels_first=True,
            clip_values=(0, 1),
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0, 1)
        )

        return mxc

    return _get_image_classifier_mx_instance


@pytest.fixture
def supported_losses_types(framework):
    def supported_losses_types():
        if framework == "keras":
            return ["label", "function_losses", "function_backend"]
        if framework == "kerastf":
            # if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
            return ["label", "function", "class"]

        raise NotImplementedError("Could not find  supported_losses_types for framework {0}".format(framework))

    return supported_losses_types


@pytest.fixture
def supported_losses_logit(framework):
    def _supported_losses_logit():
        if framework == "keras":
            return ["categorical_crossentropy_function_backend",
                    "sparse_categorical_crossentropy_function_backend"]
        if framework == "kerastf":
            # if loss_type is not "label" and loss_name not in ["categorical_hinge", "kullback_leibler_divergence"]:
            return ["categorical_crossentropy_function",
                    "categorical_crossentropy_class",
                    "sparse_categorical_crossentropy_function",
                    "sparse_categorical_crossentropy_class"]
        raise NotImplementedError("Could not find  supported_losses_logit for framework {0}".format(framework))

    return _supported_losses_logit


@pytest.fixture
def supported_losses_proba(framework):
    def _supported_losses_proba():
        if framework == "keras":
            return ["categorical_hinge_function_losses",
                    "categorical_crossentropy_label",
                    "categorical_crossentropy_function_losses",
                    "categorical_crossentropy_function_backend",
                    "sparse_categorical_crossentropy_label",
                    "sparse_categorical_crossentropy_function_losses",
                    "sparse_categorical_crossentropy_function_backend",
                    "kullback_leibler_divergence_function_losses"
                    ]
        if framework == "kerastf":
            return ["categorical_hinge_function",
                    "categorical_hinge_class",
                    "categorical_crossentropy_label",
                    "categorical_crossentropy_function",
                    "categorical_crossentropy_class",
                    "sparse_categorical_crossentropy_label",
                    "sparse_categorical_crossentropy_function",
                    "sparse_categorical_crossentropy_class",
                    "kullback_leibler_divergence_function",
                    "kullback_leibler_divergence_class"]

        raise NotImplementedError("Could not find  supported_losses_proba for framework {0}".format(framework))

    return _supported_losses_proba


@pytest.fixture
def get_image_classifier_list(framework, get_image_classifier_mx_instance):
    def _get_image_classifier_list(one_classifier=False, functional=False, **kwargs):
        sess = None
        wildcard = False
        classifier_list = None

        if kwargs.get("wildcard") is not None:
            if kwargs.get("wildcard") is True:
                wildcard = True
            del kwargs["wildcard"]

        if framework == "keras":
            if wildcard is False:
                if functional:
                    classifier_list = [get_image_classifier_kr_functional(**kwargs)]
                else:
                    classifier_list = [get_image_classifier_kr(**kwargs)]
        if framework == "tensorflow":
            if wildcard is False:
                classifier, sess = get_image_classifier_tf(**kwargs)
                classifier_list = [classifier]
        if framework == "pytorch":
            if wildcard is False:
                classifier_list = [get_image_classifier_pt(**kwargs)]
        if framework == "scikitlearn":
            logging.warning("{0} doesn't have an image classifier defined yet".format(framework))
            classifier_list = None
        if framework == "kerastf":
            if wildcard:
                classifier_list = [get_image_classifier_kr_tf_with_wildcard(**kwargs)]
            else:
                if functional:
                    classifier_list = [get_image_classifier_kr_tf_functional(**kwargs)]
                else:
                    classifier_list = [get_image_classifier_kr_tf(**kwargs)]

        if framework == "mxnet":
            if wildcard is False:
                classifier_list = [get_image_classifier_mx_instance(**kwargs)]

        if classifier_list is None:
            return None, None

        if one_classifier:
            return classifier_list[0], sess

        return classifier_list, sess

    return _get_image_classifier_list


@pytest.fixture
def get_tabular_classifier_list(framework):
    def _get_tabular_classifier_list(clipped=True):
        if framework == "keras":
            if clipped:
                classifier_list = [get_tabular_classifier_kr()]
            else:
                classifier = get_tabular_classifier_kr()
                classifier_list = [KerasClassifier(model=classifier.model, use_logits=False, channels_first=True)]

        if framework == "tensorflow":
            if clipped:
                classifier, _ = get_tabular_classifier_tf()
                classifier_list = [classifier]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(framework))
                classifier_list = None

        if framework == "pytorch":
            if clipped:
                classifier_list = [get_tabular_classifier_pt()]
            else:
                logging.warning("{0} doesn't have an uncliped classifier defined yet".format(framework))
                classifier_list = None

        if framework == "scikitlearn":
            return get_tabular_classifier_scikit_list(clipped=False)

        return classifier_list

    return _get_tabular_classifier_list


@pytest.fixture(scope="function")
def create_test_image(create_test_dir):
    test_dir = create_test_dir
    # Download one ImageNet pic for tests
    url = 'http://farm1.static.flickr.com/163/381342603_81db58bea4.jpg'
    result = requests.get(url, stream=True)
    if result.status_code == 200:
        image = result.raw.read()
        f = open(os.path.join(test_dir, 'test.jpg'), 'wb')
        f.write(image)
        f.close()

    yield os.path.join(test_dir, 'test.jpg')


@pytest.fixture(scope="session")
def framework(request):
    mlFramework = request.config.getoption("--mlFramework")
    if mlFramework not in art_supported_frameworks:
        raise Exception("mlFramework value {0} is unsupported. Please use one of these valid values: {1}".format(
            mlFramework, " ".join(art_supported_frameworks)))
    # if utils_test.is_valid_framework(mlFramework):
    #     raise Exception("The mlFramework specified was incorrect. Valid options available
    #     are {0}".format(art_supported_frameworks))
    return mlFramework


@pytest.fixture(scope="session")
def default_batch_size():
    yield 16


@pytest.fixture(scope="session")
def is_tf_version_2():
    if tf.__version__[0] == '2':
        yield True
    else:
        yield False


@pytest.fixture(scope="session")
def load_iris_dataset():
    logging.info("Loading Iris dataset")
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris), _, _ = load_dataset('iris')

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)


@pytest.fixture(scope="function")
def get_iris_dataset(load_iris_dataset, framework):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = load_iris_dataset

    x_train_iris_original = x_train_iris.copy()
    y_train_iris_original = y_train_iris.copy()
    x_test_iris_original = x_test_iris.copy()
    y_test_iris_original = y_test_iris.copy()

    yield (x_train_iris, y_train_iris), (x_test_iris, y_test_iris)

    np.testing.assert_array_almost_equal(x_train_iris_original, x_train_iris, decimal=3)
    np.testing.assert_array_almost_equal(y_train_iris_original, y_train_iris, decimal=3)
    np.testing.assert_array_almost_equal(x_test_iris_original, x_test_iris, decimal=3)
    np.testing.assert_array_almost_equal(y_test_iris_original, y_test_iris, decimal=3)


@pytest.fixture(scope="session")
def default_dataset_subset_sizes():
    n_train = 1000
    n_test = 100
    yield n_train, n_test


@pytest.fixture()
def mnist_shape(framework):
    if framework == "pytorch" or framework == "mxnet":
        return (1, 28, 28)
    else:
        return (28, 28, 1)


@pytest.fixture()
def get_default_mnist_subset(get_mnist_dataset, default_dataset_subset_sizes, mnist_shape):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train, n_test = default_dataset_subset_sizes

    x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0],) + mnist_shape).astype(np.float32)
    x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0],) + mnist_shape).astype(np.float32)

    yield (x_train_mnist[:n_train], y_train_mnist[:n_train]), (x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.fixture(scope="session")
def load_mnist_dataset():
    logging.info("Loading mnist")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), _, _ = load_dataset('mnist')
    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)


@pytest.fixture(scope="function")
def create_test_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def get_mnist_dataset(load_mnist_dataset, mnist_shape):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_dataset

    x_train_mnist = np.reshape(x_train_mnist, (x_train_mnist.shape[0],) + mnist_shape).astype(np.float32)
    x_test_mnist = np.reshape(x_test_mnist, (x_test_mnist.shape[0],) + mnist_shape).astype(np.float32)

    x_train_mnist_original = x_train_mnist.copy()
    y_train_mnist_original = y_train_mnist.copy()
    x_test_mnist_original = x_test_mnist.copy()
    y_test_mnist_original = y_test_mnist.copy()

    yield (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)

    # Check that the test data has not been modified, only catches changes in attack.generate if self has been used
    np.testing.assert_array_almost_equal(x_train_mnist_original, x_train_mnist, decimal=3)
    np.testing.assert_array_almost_equal(y_train_mnist_original, y_train_mnist, decimal=3)
    np.testing.assert_array_almost_equal(x_test_mnist_original, x_test_mnist, decimal=3)
    np.testing.assert_array_almost_equal(y_test_mnist_original, y_test_mnist, decimal=3)


# ART test fixture to skip test for specific mlFramework values
# eg: @pytest.mark.only_with_platform("tensorflow")
@pytest.fixture(autouse=True)
def only_with_platform(request, framework):
    if request.node.get_closest_marker('only_with_platform'):
        if framework not in request.node.get_closest_marker('only_with_platform').args:
            pytest.skip('skipped on this platform: {}'.format(framework))


# ART test fixture to skip test for specific mlFramework values
# eg: @pytest.mark.skipMlFramework("tensorflow","scikitlearn")
@pytest.fixture(autouse=True)
def skip_by_platform(request, framework):
    if request.node.get_closest_marker('skipMlFramework'):
        if framework in request.node.get_closest_marker('skipMlFramework').args:
            pytest.skip('skipped on this platform: {}'.format(framework))


@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record


@pytest.fixture(autouse=True)
def framework_agnostic(request, framework):
    if request.node.get_closest_marker('framework_agnostic'):
        if framework is not default_framework:
            pytest.skip('framework agnostic test skipped for framework : {}'.format(framework))
