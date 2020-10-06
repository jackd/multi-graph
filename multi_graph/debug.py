from typing import Optional

import tensorflow as tf

from multi_graph.multi_builder import MultiGraphContext
from multi_graph.tf_typing import TensorLike


class DebugBuilderContext(MultiGraphContext):
    def __init__(self, batch_size: int = 2):
        self._batch_size = batch_size
        self._model_inputs = []

    def is_pre_cache(self, x: TensorLike):
        return None

    def is_pre_batch(self, x: TensorLike):
        return None

    def is_post_batch(self, x: TensorLike):
        return None

    def pre_cache_context(self):
        return self

    def pre_batch_context(self):
        return self

    def post_batch_context(self):
        return self

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _batch(self, x: TensorLike, flat: bool = False, name: Optional[str] = None):
        if not flat:
            x = tf.expand_dims(x, axis=0)
        return tf.tile(
            x, (self.batch_size, *(1 for _ in range(x.shape.ndims - 1))), name=name,
        )

    def cache(self, x: TensorLike, name: Optional[str] = None) -> TensorLike:
        return tf.identity(x, name=name)

    def batch(self, x: TensorLike, name: Optional[str] = None):
        if isinstance(x, tf.Tensor):
            return self._batch(x, name=name)
        if isinstance(x, tf.SparseTensor):
            values = self._batch(
                x.values, flat=True, name=None if name is None else f"{name}-values",
            )
            indices = self._batch(
                x.indices, flat=True, name=None if name is None else f"{name}-indices",
            )
            b = tf.expand_dims(tf.range(self.batch_size), axis=-1)
            b = tf.tile(b, (1, tf.shape(values)[0]))
            indices = tf.concat((tf.expand_dims(b, 0), indices), axis=-1)
            dense_shape = tf.concat([(self.batch_size,), x.dense_shape], axis=0)
            return tf.SparseTensor(indices, values, dense_shape)
        if isinstance(x, tf.RaggedTensor):
            values = self._batch(x.values, flat=True)
            row_lengths = self._batch(x.row_lengths(), flat=True)
            rl2 = tf.tile(tf.expand_dims(x.nrows(), 0), self.batch_size)
            return tf.RaggedTensor.from_row_lengths(
                tf.RaggedTensor.from_row_lengths(values, row_lengths), rl2
            )
        raise TypeError(f"Invalid type for `x`: must be TensorLike, got {x}")

    def model_input(self, x: TensorLike, name: Optional[str] = None):
        assert x.shape[0] == self.batch_size
        out = tf.identity(x, name=name)
        self._model_inputs.append(out)
        return out


def debug_build_fn(build_fn, inputs, batch_size: int = 2):
    builder = DebugBuilderContext(batch_size=batch_size)
    with builder:
        args = build_fn(*inputs)
    return args
