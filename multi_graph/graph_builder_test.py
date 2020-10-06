import numpy as np
import tensorflow as tf

from multi_graph import graph_builder as gb


class GraphBuilderTest(tf.test.TestCase):
    def test_subgraph(self):
        def f(x, y0, y1):
            y = y0 + y1
            return x ** 2 + y, x + y ** 2

        graph = tf.Graph()

        with graph.as_default():
            x = tf.constant(2.0)
            y0 = tf.constant(3.0)
            y1 = tf.constant(4.0)
            z = f(x, y0, y1)
            tf.keras.layers.Dense(3)(tf.random.uniform((10, 5), dtype=tf.float32))

        fn = gb.subgraph(graph.as_graph_def(add_shapes=True), (x, y0, y1), z)
        x = tf.constant(5.0)
        y0 = tf.constant(6.0)
        y1 = tf.constant(7.0)
        actual = fn(x, y0, y1)
        expected = f(x, y0, y1)

        actual, expected = self.evaluate((actual, expected))
        self.assertEqual(len(actual), len(expected))
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])


if __name__ == "__main__":
    tf.test.main()
