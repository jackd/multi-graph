from typing import Union

import tensorflow as tf

TensorLike = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
TensorLikeSpec = Union[tf.TensorSpec, tf.RaggedTensorSpec, tf.SparseTensorSpec]
