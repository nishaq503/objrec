import os

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()


class MyLayer(Layer):
    def __init__(self, num_outputs):
        super(MyLayer, self).__init__()
        self.num_outputs = num_outputs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])

    def call(self, input_, **kwargs):
        return tf.matmul(input_, self.kernel)


if __name__ == '__main__':
    x = tf.zeros([10, 5])
    print(type(x.shape))
