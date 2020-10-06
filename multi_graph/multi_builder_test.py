import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from multi_graph import debug
from multi_graph import multi_builder as mb


def pre_cache_map(xy, label):
    return xy, label


def pre_batch_map(xy, label):
    x, y = xy
    return x * y, label


def post_batch_map(z, label):
    f = z ** 2
    return (f,), label


def model_fn(z):
    return tf.keras.layers.Dense(2, kernel_initializer="ones")(z)


def build_fn(xy, label):
    with mb.pre_cache_context():
        xy, label = pre_cache_map(xy, label)
        xy, label = tf.nest.map_structure(mb.cache, (xy, label))
    with mb.pre_batch_context():
        args = pre_batch_map(xy, label)
    args = tf.nest.map_structure(mb.batch, args)
    with mb.post_batch_context():
        features, labels = post_batch_map(*args)
    model_inp = tf.nest.map_structure(mb.model_input, features)
    model_out = model_fn(*model_inp)
    return model_out, labels


class MultiBuilderTest(tf.test.TestCase, parameterized.TestCase):
    def test_debug_builder(self):
        batch_size = 3
        x = tf.random.uniform(shape=(5,), dtype=tf.float32)
        y = tf.random.uniform(shape=(5,), dtype=tf.float32)
        x = tf.constant(x)
        y = tf.constant(y)

        label = tf.zeros((), dtype=tf.int64)
        actual_out, actual_labels = debug.debug_build_fn(
            build_fn, ((x, y), label), batch_size=batch_size
        )

        expected_out = tf.tile(tf.expand_dims(x * y, 0), (batch_size, 1))
        expected_labels = tf.zeros((batch_size,), dtype=tf.int64)
        expected_out, expected_labels = post_batch_map(expected_out, label)
        expected_out = model_fn(*expected_out)

        (actual_out, actual_labels, expected_out, expected_labels) = self.evaluate(
            (actual_out, actual_labels, expected_out, expected_labels)
        )
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_labels, expected_labels)

    @parameterized.named_parameters(("multi-model", True), ("multi-graph", False))
    def test_build_multi_graph(self, use_model_builders: bool):
        batch_size = 3
        x = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        y = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        labels = tf.random.uniform(shape=(100,), maxval=10, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices(((x, y), labels))

        # actual
        built = mb.build_multi_graph(
            build_fn, dataset.element_spec, use_model_builders=use_model_builders
        )
        processed = dataset.map(built.pre_cache_map)
        processed = processed.map(built.pre_batch_map)
        processed = processed.batch(batch_size)
        # print(processed.element_spec)
        processed = processed.map(built.post_batch_map)
        # print(processed.element_spec)
        # exit()
        actual_z = None
        actual_labels = None
        for actual_z, actual_labels in processed.take(1):
            pass
        actual_out = built.trained_model(actual_z)

        # expected
        processed = (
            dataset.map(pre_cache_map)
            .map(pre_batch_map)
            .batch(batch_size)
            .map(post_batch_map)
        )
        expected_z = None
        expected_labels = None
        for expected_z, expected_labels in processed.take(1):
            break
        expected_out = model_fn(*expected_z)

        # compare
        actual_out, actual_label, expected_out, expected_label = self.evaluate(
            (actual_out, actual_labels, expected_out, expected_labels)
        )
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_label, expected_label)


if __name__ == "__main__":
    tf.test.main()
