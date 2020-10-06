import abc
from typing import Callable, Optional

import tensorflow as tf

from multi_graph.graph_builder import GraphBuilder, GraphModelBuilder, ModelBuilder
from multi_graph.tf_typing import TensorLike, TensorLikeSpec


def get_graph(x: TensorLike) -> tf.Graph:
    if isinstance(x, tf.RaggedTensor):
        x = x.flat_values
    elif isinstance(x, tf.SparseTensor):
        x = x.values
    elif not isinstance(x, tf.Tensor):
        raise ValueError(f"Unrecognized type for x, {type(x)}")
    return x.graph


class BuiltMultiGraph:
    def __init__(
        self,
        pre_cache_map: Optional[Callable],
        pre_batch_map: Optional[Callable],
        post_batch_map: Optional[Callable],
        trained_model: Optional[tf.keras.Model],
    ):
        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map
        self._trained_model = trained_model

    @property
    def pre_cache_map(self) -> Optional[Callable]:
        return self._pre_cache_map

    @property
    def pre_batch_map(self) -> Optional[Callable]:
        return self._pre_batch_map

    @property
    def post_batch_map(self) -> Optional[Callable]:
        return self._post_batch_map

    @property
    def trained_model(self) -> Optional[tf.keras.Model]:
        return self._trained_model


class MultiGraphContext:
    _stack = []

    @staticmethod
    def get_default() -> "MultiGraphContext":
        if len(MultiGraphContext._stack) == 0:
            raise RuntimeError("No MultiGraphContext contexts open.")
        return MultiGraphContext._stack[-1]

    def __enter__(self) -> "MultiGraphContext":
        MultiGraphContext._stack.append(self)
        return self

    def __exit__(self, *args, **kwargs):
        top = MultiGraphContext._stack.pop()
        assert top is self

    @abc.abstractmethod
    def pre_cache_context(self):
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def pre_batch_context(self):
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def post_batch_context(self):
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def is_pre_cache(self, x: TensorLike) -> bool:
        """Indicate if `x` is a pre_cache tensor."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_pre_batch(self, x: TensorLike):
        """Indicate if `x` is a pre_batch tensor."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_post_batch(self, x: TensorLike):
        """Indicate if `x` is a post_batch tensor."""
        raise NotImplementedError()

    def assert_is_pre_cache(self, x: TensorLike) -> None:
        """Raise if `x` is not a pre_cache tensor."""

    def assert_is_pre_batch(self, x: TensorLike) -> None:
        """Raise if `x` is not a pre_batch tensor."""

    def assert_is_post_batch(self, x: TensorLike) -> None:
        """Raise if `x` is not a post_batch tensor."""

    def assert_is_model_tensor(self, x: TensorLike) -> None:
        """Raise if `x` is not a model tensor."""

    @abc.abstractmethod
    def cache(self, x: TensorLike, name=None) -> TensorLike:
        """Connect a tensor in the pre_cache graph to the pre_batch graph."""
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def batch(self, x: TensorLike, name=None) -> TensorLike:
        """Connect a tensor in the pre_batch graph to the post_batch graph."""
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def model_input(self, x: TensorLike, name=None) -> TensorLike:
        """Connect a tensor in the post_batch graph to the model graph."""
        raise NotImplementedError("Abstract method")


class MultiGraphBuilder(MultiGraphContext):
    def __init__(
        self, batch_size: Optional[int] = None, use_model_builders: bool = False
    ):
        constructor = GraphModelBuilder if use_model_builders else GraphBuilder
        self._pre_cache_builder = constructor(name="pre_cache")
        self._pre_batch_builder = constructor(name="pre_batch")
        self._post_batch_builder = constructor(name="post_batch")
        self._model_builder = ModelBuilder(batch_size=batch_size)
        self._batch_size = batch_size

    @property
    def pre_cache_graph(self) -> tf.Graph:
        return self._pre_cache_builder.graph

    @property
    def pre_batch_graph(self) -> tf.Graph:
        return self._pre_batch_builder.graph

    @property
    def post_batch_graph(self) -> tf.Graph:
        return self._post_batch_builder.graph

    def pre_cache_context(self):
        return self.pre_cache_graph.as_default()

    def pre_batch_context(self):
        return self.pre_batch_graph.as_default()

    def post_batch_context(self):
        return self.post_batch_graph.as_default()

    def is_pre_cache(self, x: TensorLike) -> bool:
        return hasattr(x, "graph") and x.graph is self.pre_cache_graph

    def is_pre_batch(self, x: TensorLike) -> bool:
        return hasattr(x, "graph") and x.graph is self.pre_batch_graph

    def is_post_batch(self, x: TensorLike) -> bool:
        return hasattr(x, "graph") and x.graph is self.post_batch_graph

    def assert_is_pre_cache(self, x: TensorLike) -> None:
        if get_graph(x) is not self.pre_cache_graph:
            raise ValueError("x is not part of pre_cache_graph")

    def assert_is_pre_batch(self, x: TensorLike) -> None:
        if get_graph(x) is not self.pre_batch_graph:
            raise ValueError("x is not part of pre_batch_graph")

    def assert_is_post_batch(self, x: TensorLike) -> None:
        if get_graph(x) is not self.post_batch_graph:
            raise ValueError("x is not part of post_batch_graph")

    def assert_is_model_tensor(self, x: TensorLike) -> None:
        graph = get_graph(x)
        if graph is self.pre_batch_graph:
            raise ValueError("x is part of pre_batch_graph")
        if graph is self.post_batch_graph:
            raise ValueError("x is part of post_batch_graph")

    def pre_cache_input(self, spec: TensorLikeSpec) -> tf.Tensor:
        with self.pre_cache_context():
            return self._pre_cache_builder.input(spec)

    def _cache(self, x: tf.Tensor, name=None):
        self._pre_cache_builder.add_output(x)
        assert x.shape is not None
        out = self._pre_batch_builder.input_like(x, name=name)
        return out

    def cache(self, x: TensorLike, name=None) -> TensorLike:
        self.assert_is_pre_cache(x)
        if isinstance(x, tf.Tensor):
            return self._cache(x, name=name)
        # composite tensor
        with self.pre_cache_context():
            flat_x = tf.nest.flatten(x, expand_composites=True)
        flat_x = [
            self._cache(xi, name=None if name is None else f"{name}-{i}")
            for i, xi in enumerate(flat_x)
        ]
        with self.pre_batch_context():
            out = tf.nest.pack_sequence_as(x, flat_x, expand_composites=True)
        return out

    def batch(self, x: TensorLike, name=None) -> TensorLike:
        if isinstance(x, tf.Tensor) and x.shape.ndims > 0 and x.shape[0] is None:
            raise ValueError("Cannot batch tensor with unknown first dimension")
        self._pre_batch_builder.add_output(x)
        out = self._post_batch_builder.batched_input_like(x, name=name)
        return out

    def model_input(self, x: TensorLike, name=None) -> TensorLike:
        self._post_batch_builder.add_output(x)
        return self._model_builder.input_like(x, name=name)

    def build(
        self, model_outputs, labels, weights=None, inputs_structure=None
    ) -> BuiltMultiGraph:

        rest = (labels,) if weights is None else (labels, weights)

        return BuiltMultiGraph(
            self._pre_cache_builder.build(inputs_structure=inputs_structure),
            self._pre_batch_builder.build(),
            self._post_batch_builder.build(extra_outputs=rest),
            trained_model=self._model_builder.build(model_outputs),
        )


get_default = MultiGraphContext.get_default


def _default_docs(fn):
    fn.__doc__ = getattr(MultiGraphContext, fn.__name__).__doc__
    return fn


@_default_docs
def is_pre_cache(x):
    return get_default().is_pre_cache(x)


@_default_docs
def is_pre_batch(x):
    return get_default().is_pre_batch(x)


@_default_docs
def is_post_batch(x):
    return get_default().is_post_batch(x)


@_default_docs
def pre_cache_context():
    return get_default().pre_cache_context()


@_default_docs
def pre_batch_context():
    return get_default().pre_batch_context()


@_default_docs
def post_batch_context():
    return get_default().post_batch_context()


@_default_docs
def assert_is_pre_cache(x: TensorLike):
    return get_default().assert_is_pre_cache(x)


@_default_docs
def assert_is_pre_batch(x: TensorLike):
    return get_default().assert_is_pre_batch(x)


@_default_docs
def assert_is_post_batch(x: TensorLike):
    return get_default().assert_is_post_batch(x)


@_default_docs
def assert_is_model_tensor(x: TensorLike):
    return get_default().assert_is_model_tensor(x)


@_default_docs
def cache(x: TensorLike, name=None) -> TensorLike:
    return get_default().cache(x, name=name)


@_default_docs
def batch(x: TensorLike, name=None) -> TensorLike:
    return get_default().batch(x, name=name)


@_default_docs
def model_input(x: TensorLike, name=None) -> TensorLike:
    return get_default().model_input(x, name=name)


def build_multi_graph(
    build_fn: Callable,
    inputs_spec,
    batch_size: Optional[int] = None,
    use_model_builders: bool = False,
) -> BuiltMultiGraph:
    """
    Build multiple graphs associated with build_fn.

    Args:
        build_fn: a function which takes inputs specified by `inputs_spec` and returns
            `(model_outputs, batched_labels, batched_weights?)`.
        inputs_spec: nested structure of `tf.TensorSpec`s corresponding to the unbatched
            dataset `element_spec`.
        batch_size: batch size used in builders. This gives additional shape information
            when the batch size is statically known and can lead to caching shape errors
            at build time.
        use_model_builders: se `MultiGraphBuilder.__init__`.

    Returns:
        `BuiltMultiGraph`
    """
    builder = MultiGraphBuilder(batch_size, use_model_builders=use_model_builders)
    with builder:
        inputs = tf.nest.map_structure(builder.pre_cache_input, inputs_spec)
        if isinstance(inputs, dict):
            args = build_fn(**inputs)
        elif isinstance(inputs, tf.Tensor):
            args = build_fn(inputs)
        else:
            args = build_fn(*inputs)

        if len(args) == 2:
            model_outputs, labels = args
            weights = None
        else:
            model_outputs, labels, weights = args
        return builder.build(
            model_outputs, labels, weights, inputs_structure=inputs_spec
        )
