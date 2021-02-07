import click
import numpy as np
from numpy.core.fromnumeric import shape
from tensorflow import keras
from tensorflow import data
from tensorflow.keras import layers
import fasttext


@click.group()
def cli():
    pass


def _label(filename):
    if 'klein' in filename:
        return 0
    elif 'sphere' in filename:
        return 1
    elif 'torus' in filename:
        return 2


@cli.command()
@click.argument('train')
@click.argument('validate')
@click.argument('test')
def evaluate(train, validate, test):
    """Evaluate the LSTM model.

    Train, validate, and test should be files with known file names.
    """
    # Load data.
    train_data = np.load(train)
    train_data = data.Dataset.from_tensor_slices((train_data, np.full(fill_value=_label(train), shape=(train_data.shape[0],))))

    validate_data = np.load(validate)
    validate_data = data.Dataset.from_tensor_slices((validate_data, np.full(fill_value=_label(validate), shape=(validate_data.shape[0],))))

    test_data = np.load(test)
    test_data = data.Dataset.from_tensor_slices((test_data, np.full(fill_value=_label(test), shape=(test_data.shape[0],))))

    # Build the model.
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=100, output_dim=64))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1))
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train the model.
    history = model.fit(train_data.shuffle(100).batch(64), epochs=10, verbose=1, validation_data=validate_data.batch(64))


if __name__ == "__main__":
    cli()
