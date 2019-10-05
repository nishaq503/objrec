import os

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sample_net():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    input_layer = layers.Input(shape=(28, 28), name='input')

    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)

    output_layer = layers.Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=10)
    model.evaluate(x=x_test, y=y_test)

    return


if __name__ == '__main__':
    sample_net()
