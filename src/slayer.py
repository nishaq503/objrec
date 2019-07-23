from typing import List, Tuple

import tensorflow as tf
from tensorflow.python.keras.layers import Layer


def init_params_from_args(arg, shape, default, scalar_is_valid=False):
    if isinstance(arg, (int, float)):
        if scalar_is_valid:
            return tf.fill(dims=shape, value=arg)
        else:
            raise ValueError(f'Scalar initialization values are not valid. '
                             f'Got {arg} expected Tensor of shape {shape}.')
    elif isinstance(arg, tf.Tensor):
        if arg.shape == shape:
            return arg
    elif arg is None:
        if default in (tf.random.uniform, tf.random.normal, tf.ones):
            return default(shape=shape)
        else:
            return default(shape)
    else:
        raise ValueError(f'Cannot handle parameter initialization. Got "{arg}" ')


def is_prepared_batch(input_):
    if not(isinstance(input_, tuple) and len(input_) == 4):
        return False
    else:
        batch, not_dummy_points, max_points, batch_size = input_
        return all(
            (
                isinstance(batch, tf.Tensor),
                isinstance(not_dummy_points, tf.Tensor),
                max_points > 0,
                batch_size > 0,
            )
        )


def is_list_of_tensors(input_):
    try:
        return all([isinstance(i, tf.Tensor) for i in input_])
    except TypeError:
        return False


# TODO: CHECK
def prepare_batch(
        batch: List[tf.Tensor],
        point_dim: int = None,
) -> Tuple[tf.Tensor, tf.Tensor, int, int]:
    if point_dim is None:
        point_dim = batch[0].shape(1)

    if not all([b.shape(1) == point_dim for b in batch if len(b.shape) != 0]):
        raise ValueError('Expected all inputs for SLayer to be of the same size.')

    batch_size = len(batch)
    batch_max_points = max([b.shape(0) for b in batch])

    if batch_max_points == 0:  # batch consists only of empty diagrams.
        batch_max_points = 1

    not_dummy_points = tf.zeros(shape=[batch_size, batch_max_points])
    prepared_batch = [None] * batch_size

    """
    This loop embeds each multiset in the batch to the highest dimensionality
    occurring in the batch. i.e. it zero-pads the smaller multisets. I need to
    find the tensorflow equivalent of torch.tensor.index_add_ to be able to do this. 
    """
    for i, b in enumerate(batch):
        num_points = b.shape(0)
        prepared_dgm = tf.zeros(shape=[batch_max_points, point_dim])

        if num_points > 0:
            index_selection = tf.fill(range(num_points))

            # TODO: Find tf alternative to torch.tensor.index_add_
            prepared_dgm.index_add_(0, index_selection, b)

            not_dummy_points[i, : num_points] = 1

        prepared_batch[i] = prepared_dgm

    prepared_batch = tf.stack(prepared_batch, axis=0)

    return prepared_batch, not_dummy_points, batch_max_points, batch_size


def prepare_batch_if_necessary(input_, point_dimensions=None):
    if is_prepared_batch(input_):
        return input_
    elif is_list_of_tensors(input_):
        if point_dimensions is None:
            point_dimensions = input_[0].shape(1)
        return prepare_batch(input_, point_dimensions)
    else:
        raise ValueError(f'Slayer does not recognize input format.'
                         f'Expecting "Tensor" or "prepared_batch. Got {input_}.')


class SLayerExponential(Layer):
    def __init(self, num_elements: int,
               point_dimensions: int = 3,
               centers_init: tf.Tensor = None,
               sharpness_init: tf.Tensor = None):
        super().__init__()

        self.num_elements = num_elements
        self.point_dimensions = point_dimensions

        expected_init_shape = (self.num_elements, self.point_dimensions)

        centers_init = init_params_from_args(centers_init,
                                             expected_init_shape,
                                             tf.random.uniform,
                                             scalar_is_valid=False)
        sharpness_init = init_params_from_args(sharpness_init,
                                               expected_init_shape,
                                               lambda shape: tf.ones(*shape) * 3)

        self.centers = tf.Variable(centers_init)
        self.sharpness = tf.Variable(sharpness_init)

    def forward(self, input_) -> tf.Tensor:
        temp = prepare_batch_if_necessary(input_, point_dimensions=self.point_dimensions)
        batch, not_dummy_points, max_points, batch_size = temp

        batch = tf.concat([batch] * self.num_elements, 1)

        not_dummy_points = tf.concat([not_dummy_points] * self.num_elements, 1)

        centers = tf.concat([self.centers] * max_points, 1)
        centers = tf.reshape(centers, [-1, self.point_dimensions])
        centers = tf.stack([centers] * batch_size, axis=0)

        sharpness = tf.pow(self.sharpness, 2)
        sharpness = tf.concat([sharpness] * max_points, 1)
        sharpness = tf.reshape(sharpness, [-1, self.point_dimensions])
        sharpness = tf.stack([sharpness] * batch_size, axis=0)

        x: tf.Tensor = tf.subtract(centers, batch)
        x = tf.pow(x, 2)
        x = tf.multiply(x, sharpness)
        x = tf.reduce_sum(x, axis=2)
        x = tf.exp(-x)
        x = tf.multiply(x, not_dummy_points)
        x = tf.reshape(x, [batch_size, self.num_elements, -1])
        x = tf.reduce_sum(x, axis=2)
        x = tf.squeeze(x)

        return x

    def __repr__(self):
        return f'SLayerExponential (... -> {self.n_elements} )'
