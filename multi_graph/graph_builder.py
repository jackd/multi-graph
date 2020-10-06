from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import tensorflow as tf

from multi_graph.tf_typing import TensorLike, TensorLikeSpec

T = TypeVar("T")
NameLike = Union[str, tf.Tensor]


def _spec_to_placeholder(spec, name=None):
    if isinstance(spec, tf.TensorSpec):
        return tf.keras.backend.placeholder(
            shape=spec.shape, dtype=spec.dtype, name=name
        )
    if isinstance(spec, tf.SparseTensorSpec):
        return tf.keras.backend.placeholder(
            shape=spec.shape, dtype=spec.dtype, sparse=True, name=name
        )
    if isinstance(spec, tf.RaggedTensorSpec):
        return tf.keras.backend.placeholder(
            shape=spec._shape,  # pylint:disable=protected-access
            dtype=spec._dtype,  # pylint:disable=protected-access
            ragged=True,
            name=name,
        )
    raise TypeError(f"Invalid type for spec: must be TensorSpecLike, got {spec}")


def _spec_to_input(spec, name=None):
    if isinstance(spec, tf.RaggedTensorSpec):
        shape = spec._shape  # pylint:disable=protected-access
        dtype = spec._dtype  # pylint:disable=protected-access
    else:
        shape = spec.shape
        dtype = spec.dtype
    inp = tf.keras.Input(
        shape=shape[1:],
        batch_size=shape[0],
        ragged=isinstance(spec, tf.RaggedTensorSpec),
        sparse=isinstance(spec, tf.SparseTensorSpec),
        dtype=dtype,
        name=name,
    )
    if isinstance(spec, tf.RaggedTensorSpec):
        assert isinstance(inp, tf.RaggedTensor)
    return inp


def _batched_placeholder_like(x, batch_size=None, name=None):
    shape = [batch_size, *x.shape]
    if isinstance(x, tf.RaggedTensor):
        for i in range(x.ragged_rank):
            shape[i + 2] = None

    out = tf.keras.backend.placeholder(
        shape=shape,
        dtype=x.dtype,
        sparse=isinstance(x, tf.SparseTensor),
        ragged=isinstance(x, tf.RaggedTensor),
        name=name,
    )
    if isinstance(x, tf.RaggedTensor):
        assert isinstance(out, tf.RaggedTensor)
    return out


def _placeholder_like(x, name=None):
    out = tf.keras.backend.placeholder(
        shape=x.shape,
        dtype=x.dtype,
        sparse=isinstance(x, tf.SparseTensor),
        ragged=isinstance(x, tf.RaggedTensor),
        name=name,
    )
    if isinstance(x, tf.RaggedTensor):
        assert isinstance(out, tf.RaggedTensor)
    return out


def flatten_inputs(fn, input_structure, expand_composites=False):
    """
    Change the input interface of the given function.

    Args:
        fn: function with signature `fn(*args)`.
        input_structure: different input signature
        expand_composites: used in `tf.nest.flatten`.

    Returns:
        function with signature `out_fn(inputs)`, where `inputs` must have the
            same structure as `input_structure` according to
            `tf.nest.assert_same_structure`.
    """

    def flat_fn(*inputs):
        tf.nest.assert_same_structure(
            inputs, input_structure, expand_composites=expand_composites
        )
        flat_args = tf.nest.flatten(inputs, expand_composites=expand_composites)
        return fn(*flat_args)

    return flat_fn


def repack_outputs(fn, output_structure, expand_composites=False):
    """
    Change the output interface of a given function.

    Args:
        fn: function with signature `fn(*args, **kwargs) -> Sequence`
        output_structure: desired output structure.
        expand_composites: whether outputs of `fn` have composites that should
            be reduced via `tf.nest.pack_sequence_as`.

    Returns:
        function with signature 'out_fn(*args, **kwargs) -> outupts`, where
            outputs has structure of `output_structure`.
    """

    def flat_fn(*args, **kwargs):
        out = fn(*args, **kwargs)
        return tf.nest.pack_sequence_as(
            output_structure, out, expand_composites=expand_composites
        )

    return flat_fn


def subgraph(graph_def, inputs, outputs) -> Callable:
    """
    Extract a subgraph from the given graph_def as a `@tf.function`ed callable.

    Args:
        graph_def: a `GraphDef`, like from `tf.Graph.as_graph_def`.
        inputs: tensors or op names as inputs.
        outputs: tensor or tensor names of outputs.

    Returns:
        A callable which maps f(*inputs) to a list of tensors given in outputs.
    """
    input_op_names = tuple(
        t if isinstance(t, str) else t.op.name
        for t in tf.nest.flatten(inputs, expand_composites=True)
    )
    output_names = tuple(
        t if isinstance(t, str) else t.name
        for t in tf.nest.flatten(outputs, expand_composites=True)
    )

    @tf.function()
    def graph_fn(*args, **kwargs):
        args = tf.nest.flatten((args, kwargs), expand_composites=True)
        if len(args) != len(input_op_names):
            raise ValueError(
                f"Expected {len(input_op_names)} args, got {len(args)}: {args}"
            )
        assert len(args) == len(input_op_names)
        input_map = dict(zip(input_op_names, args))
        flat_out = tf.graph_util.import_graph_def(
            graph_def, input_map=input_map, return_elements=output_names
        )
        return tf.nest.pack_sequence_as(outputs, flat_out, expand_composites=True)

    return graph_fn


class GraphBuilder:
    def __init__(self, name="graph_builder"):
        self._graph = tf.Graph()
        self._outputs = []
        self._inputs = []
        self._ctxs = []
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def graph(self):
        return self._graph

    def _validate_graph(self, x: TensorLike, name="x"):
        if isinstance(x, tf.RaggedTensor):
            self._validate_graph(x.flat_values, name=name)
        elif isinstance(x, tf.SparseTensor):
            self._validate_graph(x.indices, name=name)
        else:
            assert isinstance(x, tf.Tensor)
            if x.graph is not self._graph:
                raise ValueError("x is from a different graph - cannot add as input")

    def __enter__(self: T) -> T:
        ctx = self.graph.as_default()
        self._ctxs.append(ctx)
        ctx.__enter__()  # pylint:disable=no-member
        return self

    def __exit__(self, *args, **kwargs):
        ctx = self._ctxs.pop()
        ctx.__exit__(*args, **kwargs)

    def input(self, spec: TensorLikeSpec, name=None) -> TensorLike:
        if name is None:
            name = self.graph.unique_name(f"{self.name}/input")
        with self._graph.as_default():
            out = _spec_to_placeholder(spec, name=name)
            self._inputs.append(out)
        return out

    def input_like(self, x: TensorLike, name=None) -> TensorLike:
        if name is None:
            name = self.graph.unique_name(f"{self.name}/input")
        with self._graph.as_default():
            out = _placeholder_like(x, name=name)
        self._inputs.append(out)
        return out

    def batched_input_like(
        self, x: TensorLike, batch_size=None, name=None
    ) -> TensorLike:
        if name is None:
            name = self.graph.unique_name(f"{self.name}/input")
        with self._graph.as_default():
            out = _batched_placeholder_like(x, batch_size=batch_size, name=name)
            self._inputs.append(out)
        return out

    def add_output(self, x) -> None:
        self._validate_graph(x, "output")
        self._outputs.append(x)

    def build(
        self,
        inputs_structure=None,
        extra_outputs: Optional[Sequence[TensorLike]] = None,
    ) -> Optional[Callable]:
        inputs = self._inputs
        if inputs_structure is not None:
            inputs = tf.nest.pack_sequence_as(
                inputs_structure, inputs, expand_composites=True
            )

        outputs = self.outputs
        if extra_outputs is not None:
            for x in tf.nest.flatten(extra_outputs, expand_composites=True):
                self._validate_graph(x)
            outputs = (outputs, *extra_outputs)

        # If no outputs, return None. Maybe this should raise an error?
        if len(tf.nest.flatten(outputs, expand_composites=True)) == 0:
            return None

        return subgraph(self.graph.as_graph_def(add_shapes=True), inputs, outputs)

    @property
    def outputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._outputs)

    @property
    def inputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._inputs)


def _model_fn(
    model, inputs_structure, outputs_structure, squeezed,
):
    default_training = False  # required in tf 2.3

    def f(*inputs, training=None):
        if inputs_structure is not None:
            tf.nest.assert_same_structure(inputs, inputs_structure)
        inputs = tf.nest.flatten(inputs)
        assert len(inputs) == len(squeezed)
        inputs = [
            tf.expand_dims(inp, axis=0) if sq else inp
            for inp, sq in zip(inputs, squeezed)
        ]
        assert len(inputs) == len(model.inputs)
        outputs = model(inputs, training=training or default_training)
        if outputs_structure is not None:
            outputs = tf.nest.pack_sequence_as(outputs_structure, outputs)
        return outputs

    return f


class GraphModelBuilder(GraphBuilder):
    def __init__(self, *args, **kwargs):
        self._squeezed = []
        super().__init__(*args, **kwargs)

    def input(self, spec: TensorLikeSpec, name=None) -> TensorLike:
        with self._graph.as_default():
            if isinstance(spec, tf.RaggedTensorSpec):
                shape = spec._shape
                dtype = spec._dtype
            else:
                shape = spec.shape
                dtype = spec.dtype
            out = tf.keras.Input(
                shape=shape,
                batch_size=1,
                ragged=isinstance(spec, tf.RaggedTensorSpec),
                sparse=isinstance(spec, tf.SparseTensorSpec),
                dtype=dtype,
                name=name,
            )
            self._inputs.append(out)
            out = tf.squeeze(out, axis=0)
        self._squeezed.append(True)
        if isinstance(spec, tf.RaggedTensorSpec):
            assert isinstance(out, tf.RaggedTensor)
        elif isinstance(spec, tf.SparseTensorSpec):
            assert isinstance(out, tf.SparseTensor)
        else:
            assert isinstance(out, tf.Tensor)
        return out

    def input_like(self, x: TensorLike, name=None) -> TensorLike:
        with self._graph.as_default():
            out = tf.keras.Input(
                shape=x.shape,
                batch_size=1,
                ragged=isinstance(x, tf.RaggedTensor),
                sparse=isinstance(x, tf.SparseTensor),
                dtype=x.dtype,
                name=name,
            )
            self._inputs.append(out)
            out = tf.squeeze(out, axis=0)
        self._squeezed.append(True)
        assert type(x) is type(out)
        return out

    def batched_input_like(
        self, x: TensorLike, batch_size=None, name=None
    ) -> TensorLike:
        with self._graph.as_default():
            out = tf.keras.Input(
                shape=x.shape,
                batch_size=batch_size,
                ragged=isinstance(x, tf.RaggedTensor),
                sparse=isinstance(x, tf.SparseTensor),
                dtype=x.dtype,
                name=name,
            )
            self._inputs.append(out)
        self._squeezed.append(False)
        return out

    def build(
        self,
        inputs_structure=None,
        extra_outputs: Optional[Sequence[TensorLike]] = None,
    ):
        inputs = self._inputs
        outputs = self.outputs

        if extra_outputs is not None:
            for x in tf.nest.flatten(extra_outputs, expand_composites=True):
                self._validate_graph(x)
            outputs = (outputs, *extra_outputs)

        outputs_structure = outputs
        outputs = tf.nest.flatten(outputs)

        with self._graph.as_default():
            model = tf.keras.Model(inputs, outputs)
        if len(model.trainable_weights) != 0:
            raise RuntimeError("Model had trainable variables.")
        return _model_fn(
            model, inputs_structure, outputs_structure, squeezed=self._squeezed
        )


class ModelBuilder:
    """Similar to GraphModelBuilder, but without the graph."""

    def __init__(self, batch_size: Optional[int] = None):
        self._inputs = []
        self._outputs = []
        self._batch_size = batch_size

    def input(self, spec: TensorLikeSpec, name=None) -> TensorLike:
        out = _spec_to_input(spec, name=name)
        self._inputs.append(out)
        return out

    def input_like(self, x: TensorLike, name=None) -> TensorLike:
        out = tf.keras.Input(
            shape=x.shape[1:],
            batch_size=x.shape[0],
            ragged=isinstance(x, tf.RaggedTensor),
            sparse=isinstance(x, tf.SparseTensor),
            dtype=x.dtype,
            name=name,
        )
        self._inputs.append(out)
        return out

    def batched_input_like(self, x: TensorLike, name=None) -> TensorLike:
        out = tf.keras.Input(
            shape=x.shape,
            batch_size=self._batch_size,
            ragged=isinstance(x, tf.RaggedTensor),
            sparse=isinstance(x, tf.SparseTensor),
            dtype=x.dtype,
            name=name,
        )
        self._inputs.append(out)
        assert type(x) is type(out)
        return out

    def build(self, outputs) -> tf.keras.Model:
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            outputs = outputs[0]
        return tf.keras.Model(self._inputs, outputs)

    @property
    def outputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._outputs)

    @property
    def inputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._inputs)
