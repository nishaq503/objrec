import numpy as np
import tensorflow as tf


def tf_sq_l2_norm(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.maximum(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1), 0.0)


def tf_batch_sq_l2_norm(x: tf.Tensor) -> tf.Tensor:
    x_sq = tf.reduce_sum(tf.square(x), axis=1)
    xx, yy = tf.reshape(x_sq, shape=[-1, 1]), tf.reshape(x_sq, shape=[1, -1])
    return tf.maximum(xx + yy - 2 * tf.matmul(x, x, transpose_b=True), 0.0)


def tf_l2_norm(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.maximum(tf.sqrt(tf_sq_l2_norm(x, y)), 0.0)


def tf_batch_l2_norm(x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(tf.sqrt(tf_batch_sq_l2_norm(x)), 0.0)


def numpy_sq_l2_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum('ij,ij->i', x, x)[:, None] + np.einsum('ij,ij->i', y, y) - 2 * np.dot(x, y.T)


def numpy_l2_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.sqrt(numpy_sq_l2_norm(x, y)), 0.0)


# noinspection DuplicatedCode
def tf_calculate_distance(a: np.ndarray, b: np.ndarray, df: str) -> np.ndarray:
    """
    Calculates the distance between a and b using the distance function requested in tensorflow.

    :param a: numpy array of points.
    :param b: numpy array of points.
    :param df: distance function to use.
    :return: pairwise distances between points in a and b.
    """
    if df:  # this is a cheap way to avoid GPU usage
        return numpy_calculate_distance(a, a, df)
    
    distance_functions = {
        'sql2': tf_sq_l2_norm,
        'l2': tf_l2_norm,
    }
    
    if df in distance_functions.keys():
        distance = distance_functions[df]
    else:
        keys = ' or '.join(distance_functions.keys())
        raise ValueError(f'{df} is an invalid distance function. Possible distance functions are: {keys}.')

    a, b = np.asfarray(a), np.asfarray(b)
    distances = np.asfarray(distance(a, b))
    
    return distances


# noinspection DuplicatedCode
def tf_calculate_pairwise_distances(a: np.ndarray, df: str) -> np.ndarray:
    """
    Calculates the pairwise distances between all elements of a using the distance function requested.

    :param a: numpy array of points.
    :param df: distance function to use.
    :return: pairwise distances between points in a and b.
    """
    if df:  # this is a cheap way to avoid GPU usage
        return numpy_calculate_distance(a, a, df)
    
    distance_functions = {
        'sql2': tf_batch_sq_l2_norm,
        'l2': tf_batch_l2_norm,
    }
    
    if df in distance_functions.keys():
        distance = distance_functions[df]
    else:
        keys = ' or '.join(distance_functions.keys())
        raise ValueError(f'{df} is an invalid distance function. Possible distance functions are: {keys}.')
    
    a = np.asfarray(a)
    if a.ndim == 1:
        a = np.expand_dims(a, 0)
    
    distances = np.asfarray(distance(a))
    
    return distances


# noinspection DuplicatedCode
def numpy_calculate_distance(a: np.ndarray, b: np.ndarray, df: str) -> np.ndarray:
    """
    Calculates the distance between a and b using the distance function requested with numpy.

    :param a: numpy array of points.
    :param b: numpy array of points.
    :param df: distance function to use.
    :return: pairwise distances between points in a and b.
    """

    distance_functions = {
        'sql2': numpy_sq_l2_norm,
        'l2': numpy_l2_norm,
    }

    if df in distance_functions.keys():
        distance = distance_functions[df]
    else:
        keys = ' or '.join(distance_functions.keys())
        raise ValueError(f'{df} is an invalid distance function. Possible distance functions are: {keys}.')

    a, b = np.asfarray(a), np.asfarray(b)
    distances = np.asfarray(distance(a, b))

    return distances
