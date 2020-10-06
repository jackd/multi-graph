import functools

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

import multi_graph as mg

flags.DEFINE_integer("batch_size", default=32, help="batch size for trained model")
flags.DEFINE_bool(
    "use_model_builders",
    default=False,
    help="use model builders rather than graphs for map functions",
)


def build_fn(image: tf.Tensor, labels: tf.Tensor, num_classes: int):
    with mg.pre_cache_context():
        # cache normalized values
        image = tf.cast(image, tf.float32) / 255
    image = mg.cache(image)
    with mg.pre_batch_context():
        # add random noise
        image = image + tf.random.uniform(
            image.shape, minval=0, maxval=1, dtype=tf.float32
        )
    images = mg.batch(image)
    labels = mg.batch(mg.cache(labels))

    images = mg.model_input(images)
    x = images
    for u in (16, 32, 64):
        x = tf.keras.layers.Conv2D(u, 3, 2, padding="SAME", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    return x, labels


def main(_):
    FLAGS = flags.FLAGS
    train_ds, test_ds = tfds.load("mnist", split=("train", "test"), as_supervised=True)

    built = mg.build_multi_graph(
        functools.partial(build_fn, num_classes=10),
        train_ds.element_spec,
        batch_size=FLAGS.batch_size,
        use_model_builders=FLAGS.use_model_builders,
    )
    model = built.trained_model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )

    def map_ds(dataset):
        return (
            dataset.map(built.pre_cache_map)
            .cache()
            .map(built.pre_batch_map)
            .batch(FLAGS.batch_size)
            .map(built.post_batch_map)
        )

    train_ds, test_ds = (ds.apply(map_ds) for ds in (train_ds, test_ds))
    model.fit(train_ds, validation_data=test_ds, epochs=5)


if __name__ == "__main__":
    app.run(main)
